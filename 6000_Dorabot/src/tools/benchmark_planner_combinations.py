#!/usr/bin/env python3
"""Benchmark global/local planner combinations in headless simulator runs.

The script intentionally lives outside simulator.py so benchmark metrics can be
collected without changing the normal visual or data-collection runtime.
"""

import argparse
import csv
import itertools
import json
import multiprocessing
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[1]
os.chdir(SRC_DIR)
sys.path.insert(0, str(SRC_DIR))

from Box2D import b2ContactListener, b2PolygonShape  # noqa: E402

import simulator as simulator_module  # noqa: E402
from agents.agent import Agent  # noqa: E402
from agents.agent_state_machine import AgentState  # noqa: E402
from agents.naive_agent import NaiveAgent  # noqa: E402
from agents.sensor import Sensor, SensorContactListener  # noqa: E402
from geometry import Point  # noqa: E402
from global_planners.global_planner import GlobalPlanner  # noqa: E402
from global_planners.layered_astar_planner import LayeredAStar  # noqa: E402
from global_planners.rrtstar_planner import RRTStar  # noqa: E402
from local_planners.DD_planner import DDPlanner  # noqa: E402
from local_planners.dull_local_planner import DullPlanner  # noqa: E402
from local_planners.flc_local_planner import FLCPlanner  # noqa: E402
from local_planners.hrvo_planner import HRVOPlanner  # noqa: E402
from local_planners.hybrid_planner import HybridHRVOForcePlanner  # noqa: E402
from local_planners.local_planner import LocalPlanner  # noqa: E402
from local_planners.rvo_planner import RVOPlanner  # noqa: E402
from local_planners.virtual_force_planner import VirtualForcePlanner  # noqa: E402
from multiagent_global_planners.inash_planner import INashRRT  # noqa: E402
from multiagent_global_planners.marrtstar_planner import MARRTStar  # noqa: E402
from multiagent_global_planners.multiagent_planner import MultiAgentPlanner  # noqa: E402
from representation.gridmap_a import GridmapWithNeighbors  # noqa: E402
from server import Server  # noqa: E402
from setup_environment.loading_port import LoadingPort  # noqa: E402
from setup_environment.obstacles import Obstacles  # noqa: E402
from setup_environment.port import Port  # noqa: E402
from setup_environment.unloading_port import UnloadingPort  # noqa: E402
from setup_environment.walls import Wall  # noqa: E402
from shape import Rectangle  # noqa: E402


BUILTIN_GLOBAL_PLANNERS = {
    "LayeredAStar": LayeredAStar,
    "RRTStar": RRTStar,
    "MARRTStar": MARRTStar,
    "INashRRT": INashRRT,
}

BUILTIN_LOCAL_PLANNERS = {
    "DullPlanner": DullPlanner,
    "VirtualForcePlanner": VirtualForcePlanner,
    "RVOPlanner": RVOPlanner,
    "HRVOPlanner": HRVOPlanner,
    "DDPlanner": DDPlanner,
    "FLCPlanner": FLCPlanner,
    "HybridHRVOForcePlanner": HybridHRVOForcePlanner,
}


class BenchmarkCollisionListener(b2ContactListener):
    """Preserve sensor contact behavior while recording physical collisions."""

    def __init__(self):
        super().__init__()
        self.sensor_listener = SensorContactListener()
        self.active_pairs = set()
        self.events_total = 0
        self.events_by_kind = defaultdict(int)
        self.active_pair_steps = 0
        self.collision_steps = 0
        self.max_active_pairs = 0
        self.first_collision_step = None
        self.end_errors = 0

    def BeginContact(self, contact):
        self.sensor_listener.BeginContact(contact)
        record = self._collision_record(contact)
        if record is None:
            return
        key, kind = record
        if key not in self.active_pairs:
            self.active_pairs.add(key)
            self.events_total += 1
            self.events_by_kind[kind] += 1
            if self.first_collision_step is None:
                self.first_collision_step = simulator_module.Simulator.step_counter

    def EndContact(self, contact):
        try:
            self.sensor_listener.EndContact(contact)
        except ValueError:
            self.end_errors += 1
        record = self._collision_record(contact)
        if record is not None:
            self.active_pairs.discard(record[0])

    def sample_step(self):
        active_count = len(self.active_pairs)
        self.active_pair_steps += active_count
        self.max_active_pairs = max(self.max_active_pairs, active_count)
        if active_count:
            self.collision_steps += 1

    def as_dict(self):
        return {
            "collision_events": self.events_total,
            "collision_events_agent_agent": self.events_by_kind["agent_agent"],
            "collision_events_agent_static": self.events_by_kind["agent_static"],
            "collision_pair_steps": self.active_pair_steps,
            "collision_steps": self.collision_steps,
            "max_active_collision_pairs": self.max_active_pairs,
            "first_collision_step": self.first_collision_step,
            "contact_end_errors": self.end_errors,
        }

    def _collision_record(self, contact):
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB
        if fixture_a.sensor or fixture_b.sensor:
            return None
        obj_a = fixture_a.body.userData
        obj_b = fixture_b.body.userData
        type_a = getattr(obj_a, "type", type(obj_a).__name__)
        type_b = getattr(obj_b, "type", type(obj_b).__name__)
        if type_a != "agent" and type_b != "agent":
            return None
        key = tuple(sorted([self._object_key(obj_a), self._object_key(obj_b)]))
        kind = "agent_agent" if type_a == "agent" and type_b == "agent" else "agent_static"
        return key, kind

    def _object_key(self, obj):
        obj_type = getattr(obj, "type", type(obj).__name__)
        obj_id = getattr(obj, "id", getattr(obj, "identifier", id(obj)))
        return f"{obj_type}:{obj_id}"


def parse_csv_names(value, available, label):
    if value == "all":
        return list(available)
    names = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [name for name in names if name not in available]
    if invalid:
        valid = ", ".join(available)
        raise SystemExit(f"Unknown {label}: {', '.join(invalid)}. Valid choices: {valid}")
    return names


def create_args():
    parser = argparse.ArgumentParser(
        description="Run headless benchmarks for global/local planner combinations."
    )
    parser.add_argument(
        "--global-planners",
        default="all",
        help="Comma-separated global planner class names, or 'all'.",
    )
    parser.add_argument(
        "--local-planners",
        default="all",
        help="Comma-separated local planner class names, or 'all'.",
    )
    parser.add_argument("--minutes", type=float, default=0.5, help="Simulation minutes per combination.")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents.")
    parser.add_argument("--ports", type=int, nargs=2, default=[2, 2], metavar=("LOAD", "UNLOAD"))
    parser.add_argument("--size", type=int, nargs=2, default=[20, 20], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--speed", type=float, default=None, help="Override agent cruise speed.")
    parser.add_argument("--step", type=int, default=10, help="Simulation steps per second.")
    parser.add_argument("--obstacle", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Obstacle generation seed.")
    parser.add_argument(
        "--planner-seed",
        type=int,
        default=123,
        help="Random seed reset before each benchmark run's planning loop.",
    )
    parser.add_argument(
        "--planner-search-time",
        type=float,
        default=1.0,
        help="Cap RRT-family planner MAX_SEARCH_TIME per plan call. Use <=0 to leave code defaults.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory for CSV/JSON benchmark output, relative to src/ by default.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one combination fails.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a non-zero exit code if any combination errors or times out.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-combination and in-run progress lines.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Print benchmark progress every N real seconds. Use <=0 to disable in-run progress.",
    )
    parser.add_argument(
        "--progress-width",
        type=int,
        default=28,
        help="Character width of the progress bar.",
    )
    parser.add_argument(
        "--combo-timeout",
        type=float,
        default=120.0,
        help="Maximum real seconds allowed for one planner combination. Use <=0 to disable.",
    )
    return parser.parse_args()


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


def build_simulator_args(options, global_planner, local_planner):
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
        global_planner,
        "--lp",
        local_planner,
    ]
    if options.speed is not None:
        cmd.extend(["--speed", str(options.speed)])
    if options.obstacle is not None:
        cmd.extend(["--obstacle", options.obstacle])
    if options.seed is not None:
        cmd.extend(["--seed", str(options.seed)])
    return simulator_module.create_cmd_parser().parse_args(cmd)


def create_benchmark_simulator(options, global_planner, local_planner):
    cmd_args = simulator_module.process_cmd(build_simulator_args(options, global_planner, local_planner))
    config_data = simulator_module.json_decoder_environment_data()

    sim = simulator_module.Simulator(cmd_args)
    sim.TIME_STEP = 1.0 / config_data["simulator"]["steps_per_sec"]
    environment = sim.set_environment(cmd_args.obstacle)

    agents_number = config_data["agents"]["number"]
    agents_speed = config_data["agents"]["cruise_speed"]
    agent_angular_velocity = simulator_module.pi / config_data["agents"]["pi_divide_by_max_angular_velocity"]
    agents_dimension = config_data["agents"]["dimension"]
    workspace_width = config_data["environment"]["width_in_meters"]

    gridmap = GridmapWithNeighbors(environment.static_gridmap)
    available_index_list = gridmap.available_index_list()
    gridmap.static_obstacle_inflation()

    random.seed(500)
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

    local_cls = simulator_module.process_local_planner_cmd(local_planner)
    global_cls = simulator_module.process_global_planner_cmd(global_planner)
    if local_cls is None or global_cls is None:
        raise ValueError(f"Unable to resolve planner combination {global_planner}/{local_planner}")

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

    for port in list(sim.environment.loading_ports.values()):
        for _ in range(100):
            port.get_random_item(sim.environment.num_unloading_ports)

    listener = BenchmarkCollisionListener()
    sim.world.contactListener = listener
    random.seed(options.planner_seed)
    try:
        import numpy as np

        np.random.seed(options.planner_seed)
    except Exception:
        pass
    return sim, listener


def format_progress_bar(done, total, width):
    if total <= 0:
        return "[" + ("-" * width) + "]"
    done_width = int(width * min(done, total) / total)
    return "[" + ("#" * done_width) + ("-" * (width - done_width)) + "]"


def print_progress(label, sim, listener, step, target_steps, started_at, force=False):
    if target_steps <= 0:
        percent = 100.0
    else:
        percent = min(100.0, step / target_steps * 100.0)
    sim_time = sim.time
    pph = (sim.task_count / sim_time * 3600) if sim_time > 0 else 0.0
    bar = format_progress_bar(step, target_steps, max(4, int(getattr(sim, "progress_width", 28))))
    active_collisions = len(listener.active_pairs)
    message = (
        "{label} {bar} {percent:6.2f}% "
        "step={step}/{target_steps} sim={sim_time:.1f}s "
        "delivered={delivered} pph={pph:.2f} "
        "active_collisions={active} collision_pair_steps={pair_steps} "
        "events={events} wall={wall:.1f}s"
    ).format(
        label=label,
        bar=bar,
        percent=percent,
        step=step,
        target_steps=target_steps,
        sim_time=sim_time,
        delivered=sim.task_count,
        pph=pph,
        active=active_collisions,
        pair_steps=listener.active_pair_steps,
        events=listener.events_total,
        wall=time.time() - started_at,
    )
    print(message, flush=True)


def run_combination(options, global_planner, local_planner, progress_label=None):
    reset_global_state()
    configure_planner_runtime(options.planner_search_time)
    started_at = time.time()
    result = {
        "global_planner": global_planner,
        "local_planner": local_planner,
        "status": "ok",
        "error": "",
    }
    try:
        sim, listener = create_benchmark_simulator(options, global_planner, local_planner)
        sim.progress_width = options.progress_width
        target_steps = int(options.minutes * 60 / sim.TIME_STEP)
        last_progress_at = 0.0
        show_progress = (not options.quiet) and options.progress_interval > 0
        if show_progress:
            print_progress(progress_label or "", sim, listener, 0, target_steps, started_at)
            last_progress_at = time.time()
        for _ in range(target_steps):
            sim.step()
            listener.sample_step()
            step = simulator_module.Simulator.step_counter
            now = time.time()
            if show_progress and (now - last_progress_at >= options.progress_interval or step >= target_steps):
                print_progress(progress_label or "", sim, listener, step, target_steps, started_at)
                last_progress_at = now
        sim_time = sim.time
        throughput_pph = (sim.task_count / sim_time * 3600) if sim_time > 0 else 0.0
        result.update(
            {
                "simulation_time_sec": sim_time,
                "steps": simulator_module.Simulator.step_counter,
                "packages_delivered": sim.task_count,
                "throughput_pph": throughput_pph,
                "wall_time_sec": time.time() - started_at,
            }
        )
        result.update(listener.as_dict())
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "simulation_time_sec": 0.0,
                "steps": simulator_module.Simulator.step_counter,
                "packages_delivered": 0,
                "throughput_pph": 0.0,
                "wall_time_sec": time.time() - started_at,
                "collision_events": 0,
                "collision_events_agent_agent": 0,
                "collision_events_agent_static": 0,
                "collision_pair_steps": 0,
                "collision_steps": 0,
                "max_active_collision_pairs": 0,
                "first_collision_step": None,
                "contact_end_errors": 0,
            }
        )
    return result


def run_combination_worker(options, global_planner, local_planner, progress_label, result_queue):
    result_queue.put(run_combination(options, global_planner, local_planner, progress_label))


def timeout_result(global_planner, local_planner, timeout_sec, started_at):
    return {
        "global_planner": global_planner,
        "local_planner": local_planner,
        "status": "timeout",
        "error": "Timed out after {:.1f} real seconds".format(timeout_sec),
        "simulation_time_sec": 0.0,
        "steps": None,
        "packages_delivered": 0,
        "throughput_pph": 0.0,
        "wall_time_sec": time.time() - started_at,
        "collision_events": 0,
        "collision_events_agent_agent": 0,
        "collision_events_agent_static": 0,
        "collision_pair_steps": 0,
        "collision_steps": 0,
        "max_active_collision_pairs": 0,
        "first_collision_step": None,
        "contact_end_errors": 0,
    }


def run_combination_with_timeout(options, global_planner, local_planner, progress_label):
    if options.combo_timeout is None or options.combo_timeout <= 0:
        return run_combination(options, global_planner, local_planner, progress_label)

    started_at = time.time()
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=run_combination_worker,
        args=(options, global_planner, local_planner, progress_label, result_queue),
    )
    process.start()
    process.join(options.combo_timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        return timeout_result(global_planner, local_planner, options.combo_timeout, started_at)

    if not result_queue.empty():
        return result_queue.get()
    result = timeout_result(global_planner, local_planner, options.combo_timeout, started_at)
    result["status"] = "error"
    result["error"] = "Benchmark worker exited without returning a result"
    return result


def write_outputs(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"planner_benchmark_{timestamp}.csv"
    json_path = output_dir / f"planner_benchmark_{timestamp}.json"
    fields = [
        "global_planner",
        "local_planner",
        "status",
        "packages_delivered",
        "throughput_pph",
        "collision_events",
        "collision_events_agent_agent",
        "collision_events_agent_static",
        "collision_pair_steps",
        "collision_steps",
        "max_active_collision_pairs",
        "first_collision_step",
        "steps",
        "simulation_time_sec",
        "wall_time_sec",
        "contact_end_errors",
        "error",
    ]
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    with json_path.open("w") as file_obj:
        json.dump(results, file_obj, indent=2)
    return csv_path, json_path


def print_summary(results):
    successful = [item for item in results if item["status"] == "ok"]
    if not successful:
        print("No successful benchmark runs.", flush=True)
        return
    safest = min(
        successful,
        key=lambda item: (
            item["collision_pair_steps"],
            item["collision_events"],
            -item["throughput_pph"],
        ),
    )
    fastest = max(
        successful,
        key=lambda item: (
            item["throughput_pph"],
            -item["collision_pair_steps"],
            -item["collision_events"],
        ),
    )
    print("\nBest by fewest collisions:", flush=True)
    print(
        "  {}/{} | pair_steps={} events={} pph={:.2f} delivered={}".format(
            safest["global_planner"],
            safest["local_planner"],
            safest["collision_pair_steps"],
            safest["collision_events"],
            safest["throughput_pph"],
            safest["packages_delivered"],
        ),
        flush=True,
    )
    print("Best by throughput:", flush=True)
    print(
        "  {}/{} | pph={:.2f} delivered={} pair_steps={} events={}".format(
            fastest["global_planner"],
            fastest["local_planner"],
            fastest["throughput_pph"],
            fastest["packages_delivered"],
            fastest["collision_pair_steps"],
            fastest["collision_events"],
        ),
        flush=True,
    )


def main():
    options = create_args()
    global_names = parse_csv_names(options.global_planners, BUILTIN_GLOBAL_PLANNERS, "global planners")
    local_names = parse_csv_names(options.local_planners, BUILTIN_LOCAL_PLANNERS, "local planners")
    config_path = SRC_DIR / "config.json"
    original_config = config_path.read_text()
    results = []
    try:
        total = len(global_names) * len(local_names)
        if not options.quiet:
            print(
                "Benchmarking {} planner combinations for {} simulation minutes each.".format(
                    total, options.minutes
                ),
                flush=True,
            )
        for index, (global_planner, local_planner) in enumerate(
            itertools.product(global_names, local_names), start=1
        ):
            if not options.quiet:
                print(f"[{index}/{total}] {global_planner} + {local_planner}", flush=True)
            progress_label = f"[{index}/{total}] {global_planner}+{local_planner}"
            result = run_combination_with_timeout(options, global_planner, local_planner, progress_label)
            results.append(result)
            config_path.write_text(original_config)
            if not options.quiet:
                print(
                    "  status={status} delivered={packages_delivered} pph={throughput_pph:.2f} "
                    "collision_pair_steps={collision_pair_steps} events={collision_events}".format(
                        **result
                    ),
                    flush=True,
                )
                if result["status"] != "ok":
                    print(f"  error={result['error']}", flush=True)
            if options.stop_on_error and result["status"] != "ok":
                break
    finally:
        config_path.write_text(original_config)

    csv_path, json_path = write_outputs(results, options.output_dir)
    print_summary(results)
    print(f"\nCSV: {csv_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    errors = [item for item in results if item["status"] != "ok"]
    if errors:
        print(f"Failed combinations: {len(errors)}", flush=True)
    if errors and options.fail_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
