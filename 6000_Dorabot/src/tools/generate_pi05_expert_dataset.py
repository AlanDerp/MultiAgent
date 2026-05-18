#!/usr/bin/env python3
"""Generate task-aware joint BC expert data for centralized pi05 training."""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[1]
os.chdir(SRC_DIR)
sys.path.insert(0, str(SRC_DIR))

from Box2D import b2PolygonShape  # noqa: E402

import simulator as simulator_module  # noqa: E402
from agents.agent import Agent  # noqa: E402
from agents.agent_state_machine import AgentState  # noqa: E402
from agents.naive_agent import NaiveAgent  # noqa: E402
from agents.sensor import Sensor, SensorContactListener  # noqa: E402
from data_logger import DataLogger  # noqa: E402
from geometry import Point  # noqa: E402
from global_planners.layered_astar_planner import LayeredAStar  # noqa: E402
from global_planners.rrtstar_planner import RRTStar  # noqa: E402
from local_planners.DD_planner import DDPlanner  # noqa: E402
from local_planners.dull_local_planner import DullPlanner  # noqa: E402
from local_planners.flc_local_planner import FLCPlanner  # noqa: E402
from local_planners.hrvo_planner import HRVOPlanner  # noqa: E402
from local_planners.hybrid_planner import HybridHRVOForcePlanner  # noqa: E402
from local_planners.rvo_planner import RVOPlanner  # noqa: E402
from local_planners.virtual_force_planner import VirtualForcePlanner  # noqa: E402
from multiagent_global_planners.inash_planner import INashRRT  # noqa: E402
from multiagent_global_planners.marrtstar_planner import MARRTStar  # noqa: E402
from representation.gridmap_a import GridmapWithNeighbors  # noqa: E402
from server import Server  # noqa: E402
from setup_environment.loading_port import LoadingPort  # noqa: E402
from setup_environment.obstacles import Obstacles  # noqa: E402
from setup_environment.port import Port  # noqa: E402
from setup_environment.unloading_port import UnloadingPort  # noqa: E402
from setup_environment.walls import Wall  # noqa: E402
from shape import Rectangle  # noqa: E402


GLOBAL_PLANNERS = {
    "LayeredAStar": LayeredAStar,
    "RRTStar": RRTStar,
    "MARRTStar": MARRTStar,
    "INashRRT": INashRRT,
}

LOCAL_PLANNERS = {
    "DullPlanner": DullPlanner,
    "VirtualForcePlanner": VirtualForcePlanner,
    "RVOPlanner": RVOPlanner,
    "HRVOPlanner": HRVOPlanner,
    "DDPlanner": DDPlanner,
    "FLCPlanner": FLCPlanner,
    "HybridHRVOForcePlanner": HybridHRVOForcePlanner,
}

DEFAULT_MIX = "RRTStar:VirtualForcePlanner:8,RRTStar:RVOPlanner:2"


@dataclass(frozen=True)
class EpisodeSpec:
    global_planner: str
    local_planner: str
    episode_index: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate centralized joint_bc_v1 expert samples for pi05 fine-tuning."
    )
    parser.add_argument("--mix", default=DEFAULT_MIX, help="Planner mix as GP:LP:COUNT entries.")
    parser.add_argument("--minutes", type=float, default=5.0, help="Simulation minutes per episode.")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--ports", type=int, nargs=2, default=[2, 2], metavar=("LOAD", "UNLOAD"))
    parser.add_argument("--size", type=int, nargs=2, default=[20, 20], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--step", type=int, default=30, help="Simulation steps per second.")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--obstacle", type=str, default=None)
    parser.add_argument("--base-seed", type=int, default=20260517)
    parser.add_argument("--planner-search-time", type=float, default=1.0)
    parser.add_argument("--record-freq", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=30.0)
    return parser.parse_args()


def parse_mix(value):
    specs = []
    for entry in [item.strip() for item in value.split(",") if item.strip()]:
        parts = entry.split(":")
        if len(parts) != 3:
            raise SystemExit(f"Invalid mix entry `{entry}`. Expected GP:LP:COUNT.")
        global_name, local_name, count_text = parts
        if global_name not in GLOBAL_PLANNERS:
            raise SystemExit(f"Unknown global planner `{global_name}`.")
        if local_name not in LOCAL_PLANNERS:
            raise SystemExit(f"Unknown local planner `{local_name}`.")
        count = int(count_text)
        if count <= 0:
            raise SystemExit(f"Planner count must be positive in `{entry}`.")
        for _ in range(count):
            specs.append(EpisodeSpec(global_name, local_name, len(specs)))
    return specs


def default_output_path():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("data_samples") / f"jointbc_taskaware_rrt_mix_quick_{timestamp}.jsonl"


def reset_global_state():
    simulator_module.Simulator.step_counter = 0
    Port.simulator_step = 0
    Agent.counter = 0
    Obstacles.counter = 0
    Wall.counter = 0
    LoadingPort.counter = 0
    UnloadingPort.counter = 0


def configure_planner_runtime(search_time):
    MARRTStar.VISUAL = False
    INashRRT.VISUAL = False
    RRTStar.VISUAL = False
    MARRTStar.report_time = 5
    INashRRT.report_time = 10
    if search_time is not None and search_time > 0:
        RRTStar.MAX_SEARCH_TIME = search_time
        MARRTStar.MAX_SEARCH_TIME = search_time
        INashRRT.MAX_SEARCH_TIME = search_time


def build_simulator_args(options, spec, obstacle_seed):
    cmd = [
        "--default",
        "-t",
        str(options.minutes),
        "--agent",
        str(options.agents),
        "--port",
        str(options.ports[0]),
        str(options.ports[1]),
        "--size",
        str(options.size[0]),
        str(options.size[1]),
        "--resolution",
        str(options.resolution),
        "--step",
        str(options.step),
        "--gp",
        spec.global_planner,
        "--lp",
        spec.local_planner,
    ]
    if options.speed is not None:
        cmd.extend(["--speed", str(options.speed)])
    if options.obstacle is not None:
        cmd.extend(["--obstacle", options.obstacle, "--seed", str(obstacle_seed)])
    return simulator_module.create_cmd_parser().parse_args(cmd)


def create_simulator(options, spec, output_path, seeds):
    cmd_args = simulator_module.process_cmd(build_simulator_args(options, spec, seeds["obstacle"]))
    config_data = simulator_module.json_decoder_environment_data()

    sim = simulator_module.Simulator(cmd_args)
    sim.TIME_STEP = 1.0 / config_data["simulator"]["steps_per_sec"]
    sim.record_data = True
    sim.data_logger = DataLogger(
        frequency=options.record_freq,
        filename=str(output_path),
        mode="joint_bc",
        include_global_state=True,
    )
    environment = sim.set_environment(cmd_args.obstacle)

    agents_number = config_data["agents"]["number"]
    agents_speed = config_data["agents"]["cruise_speed"]
    agent_angular_velocity = simulator_module.pi / config_data["agents"]["pi_divide_by_max_angular_velocity"]
    agents_dimension = config_data["agents"]["dimension"]
    workspace_width = config_data["environment"]["width_in_meters"]

    gridmap = GridmapWithNeighbors(environment.static_gridmap)
    available_index_list = gridmap.available_index_list()
    gridmap.static_obstacle_inflation()

    random.seed(seeds["spawn"])
    spawn_locations = random.sample(available_index_list, agents_number)
    agents = []
    for num in range(agents_number):
        row = int(spawn_locations[num] / workspace_width)
        col = spawn_locations[num] % workspace_width
        agent = NaiveAgent(
            shape=Rectangle(col, row, agents_dimension, agents_dimension),
            position=Point(col, row),
            speed=agents_speed,
            angular_velocity=agent_angular_velocity,
        )
        agent.angularVelocity = 1
        agent.sensor = Sensor(
            radius=5,
            angle=simulator_module.pi,
            location=agent.shape.get_box2d_location(),
            shape=b2PolygonShape,
        )
        if sim.free_control:
            agent.state = AgentState.CRUISE
        agents.append(agent)

    server = Server(environment=environment, agents=agents)
    sim.server = server

    local_cls = LOCAL_PLANNERS[spec.local_planner]
    global_cls = GLOBAL_PLANNERS[spec.global_planner]
    for agent in agents:
        agent.connect_to_central_server(server)
        sim.set_local_planner(agent, local_cls)
        sim.set_global_planner(agent, global_cls)

    sim.set_agents(agents)
    if not sim.free_control:
        sim.set_ports(environment.unloading_ports)
        sim.set_ports(environment.loading_ports)
        sim.loading_ports = list(environment.loading_ports.values())
        sim.unloading_ports = list(environment.unloading_ports.values())
    sim.set_walls(environment.walls)
    sim.set_obstacles(environment.obstacles)
    sim.ini_perception()

    random.seed(seeds["task"])
    for port in list(sim.environment.loading_ports.values()):
        for _ in range(100):
            port.get_random_item(sim.environment.num_unloading_ports)

    sim.world.contactListener = SensorContactListener()
    random.seed(seeds["planner"])
    try:
        import numpy as np

        np.random.seed(seeds["planner"])
    except Exception:
        pass
    return sim


def run_episode(options, spec, output_path):
    reset_global_state()
    configure_planner_runtime(options.planner_search_time)
    seed_base = options.base_seed + spec.episode_index * 9973
    seeds = {
        "spawn": seed_base + 11,
        "task": seed_base + 23,
        "planner": seed_base + 37,
        "obstacle": seed_base + 53,
    }
    sim = create_simulator(options, spec, output_path, seeds)
    target_steps = int(options.minutes * 60 / sim.TIME_STEP)
    started_at = time.time()
    last_progress_at = started_at
    for _ in range(target_steps):
        sim.step()
        now = time.time()
        if options.progress_interval > 0 and now - last_progress_at >= options.progress_interval:
            print(
                "episode={idx} {gp}+{lp} step={step}/{target} delivered={delivered} wall={wall:.1f}s".format(
                    idx=spec.episode_index + 1,
                    gp=spec.global_planner,
                    lp=spec.local_planner,
                    step=simulator_module.Simulator.step_counter,
                    target=target_steps,
                    delivered=sim.task_count,
                    wall=now - started_at,
                ),
                flush=True,
            )
            last_progress_at = now
    sim.data_logger.export_data()
    return {
        "episode": spec.episode_index,
        "global_planner": spec.global_planner,
        "local_planner": spec.local_planner,
        "steps": simulator_module.Simulator.step_counter,
        "simulation_time_sec": sim.time,
        "packages_delivered": sim.task_count,
        "seeds": seeds,
    }


def main():
    options = parse_args()
    specs = parse_mix(options.mix)
    output_path = options.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not options.overwrite:
        raise SystemExit(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()

    config_path = SRC_DIR / "config.json"
    original_config = config_path.read_text()
    summary_path = output_path.with_suffix(".summary.json")
    results = []
    try:
        for spec in specs:
            print(
                "[{}/{}] generating {}+{} -> {}".format(
                    spec.episode_index + 1,
                    len(specs),
                    spec.global_planner,
                    spec.local_planner,
                    output_path,
                ),
                flush=True,
            )
            result = run_episode(options, spec, output_path)
            results.append(result)
            config_path.write_text(original_config)
            print(
                "  delivered={packages_delivered} sim={simulation_time_sec:.1f}s steps={steps}".format(
                    **result
                ),
                flush=True,
            )
    finally:
        config_path.write_text(original_config)

    summary = {
        "output": str(output_path),
        "mix": options.mix,
        "minutes": options.minutes,
        "record_freq": options.record_freq,
        "episodes": len(results),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
