#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piplan.data.lerobot_writer import LeRobotDatasetWriter
from piplan.legacy_expert.dorabot_log_reader import DorabotLogReader
from piplan.legacy_expert.expert_dataset_converter import ExpertDatasetConverter


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert legacy Dorabot expert JSONL to LeRobot dataset.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default="local/piplan-dorabot-expert")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action-source", choices=["expert", "executed", "safe", "proposed"], default="expert")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    rows = DorabotLogReader().read(args.input)
    examples = ExpertDatasetConverter(include_images=not args.no_images, action_source=args.action_source).convert(rows)
    if not examples:
        raise SystemExit("No convertible expert examples found.")
    LeRobotDatasetWriter().write(output, examples, repo_id=args.repo_id, fps=args.fps, include_images=not args.no_images, append=args.append)
    print(f"converted {len(examples)} frames into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
