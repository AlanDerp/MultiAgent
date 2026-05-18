#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from piplan.evaluation.rollout import run_rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "pi05"], default="mock")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model_id", default="lerobot/pi05_base")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"])
    parser.add_argument("--lerobot_path", default="../lerobot/src")
    parser.add_argument("--save-video", default=None)
    args = parser.parse_args()
    run_rollout(
        mode=args.mode,
        steps=args.steps,
        agents=args.agents,
        config_path=args.config,
        model_id=args.model_id,
        device=args.device,
        lerobot_path=args.lerobot_path,
        save_video=args.save_video,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
