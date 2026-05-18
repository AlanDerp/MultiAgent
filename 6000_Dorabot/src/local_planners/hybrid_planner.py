from .local_planner import LocalPlanner
from geometry import Point, Vector, compute_direction
import math
# 假设这些辅助函数在你的 rvo_planner 中
from .rvo_planner import compute_V_des, compute_rvo_BA, norm
# 引入你刚才代码里的 HRVO 计算核心
from .hrvo_planner import compute_hrvo_DD_BA, intersect, compute_angular_velocity

class HybridHRVOForcePlanner(LocalPlanner):
    """
    融合了 HRVO 的多智能体协作能力与虚拟力场的平滑性，解决死锁和抽搐问题。
    """
    def __init__(self, agent):
        super().__init__(agent)
        self.smooth_weight = 0.3  # 虚拟力所占权重
        self.repel_constant = 15.0 # 斥力常数

    def compute_plan(self, position, velocity, environment, sensor_observation, global_planner_path):
        if self.agent.destination_location is None:
            return (0.0, 0.0)

        # --- 1. 获取 HRVO 建议速度 ---
        pA = position
        vA = velocity
        vA_des = compute_V_des(self.agent, global_planner_path)
        rA = self.agent.shape.get_radius()
        
        RVO_BA_all = []

        # 处理 Agent（使用 HRVO 逻辑）
        for agentB in sensor_observation.other_agents_state_in_range_of(3):
            self.agent.potential_collision = True
            # 如果对方没有目标或是排队状态，视为静态障碍物
            if agentB.userData.destination_location is None:
                RVO_BA = compute_rvo_BA(pA, vA, rA, agentB.position, agentB.linearVelocity, agentB.userData.shape.get_radius(), reciprocal=False)
            else:
                RVO_BA = compute_hrvo_DD_BA(self.agent, agentB.userData)
            RVO_BA_all.append(RVO_BA)

        # 处理静态障碍物（利用 RVO 逻辑构建硬约束边界）
        for obstacle in sensor_observation.ports_in_range_of(2):
            pB = Point(obstacle.userData.location.x + obstacle.userData.dimension[0]*0.5, 
                       obstacle.userData.location.y + obstacle.userData.dimension[1]*0.5)
            rB = math.sqrt(obstacle.userData.dimension[0]**2 + obstacle.userData.dimension[1]**2) / 2
            VO_BA = compute_rvo_BA(pA, vA, rA, pB, [0, 0], rB, reciprocal=False)
            RVO_BA_all.append(VO_BA)

        # 通过几何求交获取 HRVO 速度结果
        if RVO_BA_all:
            hrvo_vel = Vector(*intersect(self.agent, RVO_BA_all))
        else:
            hrvo_vel = Vector(*vA_des)

        # --- 2. 引入虚拟力修正 (Virtual Force Injection) ---
        # 我们计算一个总的斥力，用来平滑 HRVO 的输出
        total_repel_force = Vector(0, 0)
        
        # 针对近距离障碍物产生斥力，防止靠墙抽搐
        for body in sensor_observation.ports_in_range_of(1.5): # 距离更近时触发
            obs = body.userData
            pB = Point(obs.location.x + obs.dimension[0]*0.5, obs.location.y + obs.dimension[1]*0.5)
            dist = pA.distance(pB)
            if dist > 0.1:
                # 经典的 Virtual Force 斥力公式
                direction_vec = compute_direction(pA, pB)
                force_mag = -self.repel_constant / (dist**3) # 相比 dist**4 稍微平滑点
                total_repel_force = total_repel_force + Vector(direction_vec[0], direction_vec[1]).scale(force_mag)

        # --- 3. 结果合成 ---
        # 最终速度 = HRVO 引导速度 + 虚拟斥力修正
        # 这样即使 HRVO 陷入死锁，斥力也会把机器人推开
        final_vel_vec = hrvo_vel + total_repel_force
        
        # 归一化并限速
        current_speed = math.sqrt(final_vel_vec.x**2 + final_vel_vec.y**2)
        if current_speed > 0.001:
            scale = min(self.agent.cruise_speed, current_speed) / current_speed
            final_vel = (final_vel_vec.x * scale, final_vel_vec.y * scale)
        else:
            final_vel = (0.0, 0.0)

        # 如果你的机器人是差分驱动，可以在这里保留 angular velocity 的转化逻辑
        # 这里为了符合 TemplateLocal 格式，返回 (vx, vy)
        return final_vel