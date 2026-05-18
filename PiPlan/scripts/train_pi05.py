#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


"""PiPlan pi0.5 training launcher.

Two-stage strategy:

Stage 1: SFT
  - Data: data/expert/lerobot/dorabot_joint, produced from offline expert
    trajectories. The expert source can be CBS/PIBT or legacy Dorabot
    RRT/VirtualForce data converted into the same joint_bc_v1-compatible
    LeRobot layout.
  - Framework: LeRobot trainer with pi0.5 policy.
  - Fine-tuning: use LoRA or equivalent parameter-efficient adapters when
    available in the local LeRobot checkout.
  - Freezing: freeze early vision encoder layers first; keep action expert and
    projection layers trainable.
  - Loss: MSE over continuous joint actions [vx0, vy0, ..., vxN, vyN], or CE if
    a later experiment discretizes the velocity bins.

Stage 2: RLFT placeholder
  - Algorithm candidates: GRPO or REINFORCE++.
  - Reward: use the immediate reward defined in piplan/data/logger.py.
  - Environment: wrap PiPlan simulator as a gym-like environment exposing
    centralized observation and joint action chunk APIs.
  - This script intentionally does not implement RLFT yet.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Thin LeRobot pi0.5 training launcher.")
    parser.add_argument("--dataset-repo-id", default="local/dorabot-pi05-joint-taskaware-quick")
    parser.add_argument(
        "--dataset-root",
        default="../6000_Dorabot/src/converted/jointbc_taskaware_rrt_mix_quick_lerobot",
    )
    parser.add_argument("--output-dir", default="outputs/train/piplan_pi05_taskaware_smoke")
    parser.add_argument("--policy-pretrained-path", default="lerobot/pi05_base")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--local-pretrained-dir", default="outputs/cache/pi05_base_local_tokenizer")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-compile-model", action="store_true")
    args = parser.parse_args()
    pretrained_path = _prepare_local_pretrained_path(
        args.policy_pretrained_path,
        tokenizer_path=args.tokenizer_path,
        local_dir=args.local_pretrained_dir,
    )
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={Path(args.dataset_root).expanduser().resolve()}",
        "--policy.type=pi05",
        f"--output_dir={args.output_dir}",
        "--job_name=piplan_pi05",
        f"--policy.pretrained_path={pretrained_path}",
        "--policy.max_state_dim=163",
        f"--policy.compile_model={str(not args.no_compile_model).lower()}",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        "--policy.freeze_vision_encoder=false",
        "--policy.train_expert_only=false",
        "--policy.push_to_hub=false",
        f"--steps={args.steps}",
        f"--policy.device={args.device}",
        f"--batch_size={args.batch_size}",
    ]
    env = os.environ.copy()
    env.setdefault("HF_DATASETS_CACHE", "/private/tmp/piplan_hf_datasets_cache")
    return subprocess.run(cmd, check=True, env=env).returncode


def _prepare_local_pretrained_path(pretrained_path: str, tokenizer_path: str | None, local_dir: str) -> str:
    if pretrained_path != "lerobot/pi05_base" and not Path(pretrained_path).is_dir():
        return pretrained_path
    tokenizer = Path(tokenizer_path).expanduser().resolve() if tokenizer_path else _find_hf_snapshot(
        "models--google--paligemma-3b-pt-224",
        required=("config.json", "tokenizer.json", "tokenizer_config.json"),
    )
    base = Path(pretrained_path).expanduser().resolve() if Path(pretrained_path).is_dir() else _find_hf_snapshot(
        "models--lerobot--pi05_base",
        required=("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"),
    )
    if tokenizer is None or base is None:
        return pretrained_path

    target = Path(local_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "model.safetensors", "policy_postprocessor.json"]:
        dst = target / name
        if not dst.exists():
            dst.symlink_to(base / name)

    preprocessor = json.loads((base / "policy_preprocessor.json").read_text())
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            step.setdefault("config", {})["tokenizer_name"] = str(tokenizer)
    (target / "policy_preprocessor.json").write_text(json.dumps(preprocessor, indent=2) + "\n")
    return str(target)


def _find_hf_snapshot(cache_name: str, required: tuple[str, ...]) -> Path | None:
    root = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    if not root.exists():
        return None
    for snapshot in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if snapshot.is_dir() and all((snapshot / name).exists() for name in required):
            return snapshot
    return None


if __name__ == "__main__":
    raise SystemExit(main())
