"""
用于测试 π₀.₅（pi05）模型推理的脚本（无需真实机器人硬件）。

特点：
- 严格复用 LeRobot 的“配置 + processor”方式，不手写 tokenizer / key 映射。
- 支持 `select_action`（单步）和 `predict_action_chunk`（多步 chunk）。

安装（仓库根目录）：
  pip install -e ".[pi]"

运行：
  python testpi05.py --model_id lerobot/pi05_base
"""

from __future__ import annotations

import argparse
import sys


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError as e:
        print("未检测到 PyTorch（torch）。请先在你的 Python 环境安装它。")
        print("例如：")
        print("  - CPU:  pip install torch")
        print("  - CUDA: 按你的 CUDA 版本参考 PyTorch 官方安装命令")
        print(e)
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="lerobot/pi05_base")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])
    parser.add_argument("--task", type=str, default="Pick up the red block and place it in the bin")
    parser.add_argument("--action_dim", type=int, default=None, help="可选：覆盖 mock action 维度（默认用模型 config）")
    parser.add_argument("--state_dim", type=int, default=None, help="可选：覆盖 mock state 维度（默认用模型 config）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device_str = args.device or _auto_device()
    device = torch.device(device_str)

    try:
        import lerobot  # noqa: F401
    except ImportError as e:
        print('lerobot 未安装，请在仓库根目录运行：pip install -e ".[pi]"')
        print(e)
        return 1

    # 1) 加载 policy（从 Hub 或本地目录）
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE

    print(f"Loading policy from `{args.model_id}` ...")
    policy = PI05Policy.from_pretrained(args.model_id, strict=True)
    policy.eval()

    # 2) 加载/构建同款 processors（优先从 pretrained 里读取 pipeline 配置）
    #    这里 override device_processor，确保 processors 的 device 和你当前选择一致
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 3) 构造一份 mock batch（走 processor 后才会变成 model 需要的 tokens/masks）
    image_keys = list(policy.config.image_features.keys())
    if len(image_keys) == 0:
        raise ValueError("pi05 需要至少一个视觉输入（image feature），但 config.image_features 为空。")

    inferred_state_dim = (
        args.state_dim
        or (policy.config.input_features.get(OBS_STATE).shape[0] if policy.config.input_features else None)
        or policy.config.max_state_dim
    )
    inferred_action_dim = (
        args.action_dim
        or (policy.config.output_features.get(ACTION).shape[0] if policy.config.output_features else None)
        or policy.config.max_action_dim
    )

    batch = {
        OBS_STATE: torch.randn(inferred_state_dim, dtype=torch.float32),
        # 预处理里会加 batch 维度；这里保持“单条样本”即可
        "task": args.task,
    }

    # images 用 [0, 1] float32，符合 LeRobot 约定；pi05 内部会转到 [-1, 1] 并 resize/pad
    for k in image_keys:
        batch[k] = torch.rand(3, 224, 224, dtype=torch.float32)

    # 4) 推理
    batch_proc = preprocessor(batch)
    with torch.no_grad():
        action_1 = policy.select_action(batch_proc)
        action_1 = postprocessor(action_1)

        chunk = policy.predict_action_chunk(batch_proc)
        chunk = postprocessor(chunk)

    #region
    print("Done.")
    print(f"- device: {device_str}")
    print(f"- image keys: {image_keys}")
    print(f"- state dim (mock): {inferred_state_dim}")
    print(f"- action dim (mock): {inferred_action_dim}")
    print(f"- select_action: shape={tuple(action_1.shape)}  range=({action_1.min().item():.4f}, {action_1.max().item():.4f})")
    print(f"- predict_action_chunk: shape={tuple(chunk.shape)}")
    #endregion

    return 0


if __name__ == "__main__":
    raise SystemExit(main())