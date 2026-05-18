#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piplan.legacy_expert.dorabot_runner import LegacyDorabotRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect legacy 6000_Dorabot expert data.")
    parser.add_argument("--dorabot-root", default="../6000_Dorabot")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minutes", type=float, default=1.0)
    parser.add_argument("--gp", default="RRTStar")
    parser.add_argument("--lp", default="VirtualForcePlanner")
    parser.add_argument("--record-mode", default="joint_bc", choices=["bc", "joint_bc"])
    parser.add_argument("--record-freq", type=int, default=10)
    args = parser.parse_args()
    LegacyDorabotRunner(args.dorabot_root).collect(
        output=args.output,
        minutes=args.minutes,
        global_planner=args.gp,
        local_planner=args.lp,
        record_mode=args.record_mode,
        record_freq=args.record_freq,
        include_global_state=True,
    )
    print(f"expert data saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
