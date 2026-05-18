from __future__ import annotations

import subprocess
from pathlib import Path


class LegacyDorabotRunner:
    """Run 6000_Dorabot as an isolated expert-data subprocess."""

    def __init__(self, dorabot_root: str | Path):
        self.dorabot_root = Path(dorabot_root).resolve()

    def collect(
        self,
        output: str | Path,
        minutes: float = 1.0,
        global_planner: str = "RRTStar",
        local_planner: str = "VirtualForcePlanner",
        record_mode: str = "joint_bc",
        record_freq: int = 10,
        include_global_state: bool = True,
        extra_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess:
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python",
            "src/simulator.py",
            "-t",
            str(minutes),
            "--gp",
            global_planner,
            "--lp",
            local_planner,
            "--record",
            "--record_mode",
            record_mode,
            "--record_freq",
            str(record_freq),
            "--record_file",
            str(output),
        ]
        if include_global_state:
            cmd.append("--record_global_state")
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, cwd=self.dorabot_root, check=True)
