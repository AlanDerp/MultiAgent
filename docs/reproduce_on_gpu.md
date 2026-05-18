# Reproduce Dorabot/PiPlan Training On A GPU Machine

This repo intentionally does not commit generated expert samples, converted LeRobot datasets, checkpoints, or model weights. Regenerate them on the target machine.

## 1. Create the environment

Conda path:

```bash
conda env create -f environment-dorabot.yml
conda activate dorabot
python -m pip install -e ./lerobot
```

`environment-dorabot.yml` is exported from the local macOS `dorabot` env. On a Linux NVIDIA host, if the solver fails on macOS-only packages such as `pyobjc-*` or `gnureadline`, remove those entries and reinstall PyTorch with the CUDA wheel matching the machine. The official selector is <https://pytorch.org/get-started/locally/>.

Pip fallback:

```bash
python -m pip install -r requirements-dorabot.txt
python -m pip install -e ./lerobot
```

For an NVIDIA machine, verify CUDA after installing the matching PyTorch build for that machine. For example, current PyTorch Linux pip wheels list CUDA 11.8, 12.6, and 12.8 options; choose the one that matches the target driver/runtime:

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 2. Generate expert JSONL samples

From the repo root:

```bash
cd 6000_Dorabot/src
python tools/generate_pi05_expert_dataset.py \
  --mix RRTStar:VirtualForcePlanner:80,RRTStar:RVOPlanner:20 \
  --minutes 5 \
  --agents 10 \
  --record-freq 1 \
  --planner-search-time 1.0 \
  --progress-interval 30 \
  --output data_samples/jointbc_taskaware_rrt_mix_gpu.jsonl \
  --overwrite
```

For a quick pipeline smoke test, reduce the mix and duration:

```bash
python tools/generate_pi05_expert_dataset.py \
  --mix RRTStar:VirtualForcePlanner:1 \
  --minutes 0.2 \
  --agents 2 \
  --record-freq 1 \
  --output data_samples/jointbc_taskaware_smoke.jsonl \
  --overwrite
```

## 3. Convert JSONL to LeRobot format

From `6000_Dorabot/src`:

```bash
python tools/convert_joint_bc_lerobot_stream.py \
  --input "data_samples/jointbc_taskaware_rrt_mix_gpu.jsonl" \
  --output converted/jointbc_taskaware_rrt_mix_gpu_lerobot \
  --repo_id local/dorabot-pi05-joint-taskaware-gpu \
  --pad_state_dim 163 \
  --pad_action_dim 20 \
  --progress-interval 10000 \
  --overwrite
```

Add `--include_images` if you want the `observation.images.topdown` frames in the dataset. The current training launcher uses state/action dimensions `163 -> 20`.

## 4. Start pi0.5 training

From the repo root:

```bash
cd PiPlan
python scripts/train_pi05.py \
  --dataset-repo-id local/dorabot-pi05-joint-taskaware-gpu \
  --dataset-root ../6000_Dorabot/src/converted/jointbc_taskaware_rrt_mix_gpu_lerobot \
  --output-dir outputs/train/piplan_pi05_taskaware_gpu \
  --policy-pretrained-path lerobot/pi05_base \
  --steps 20000 \
  --batch-size 8 \
  --device cuda \
  --no-compile-model
```

If the target machine already has a local pi0.5 checkpoint or Hugging Face cache, pass it through `--policy-pretrained-path`. The launcher sets `policy.push_to_hub=false`, so training stays local by default.

## 5. Run the trained model

```bash
cd PiPlan
python scripts/run_pi05_sim.py \
  --mode pi05 \
  --model_id outputs/train/piplan_pi05_taskaware_gpu/checkpoints/020000/pretrained_model \
  --lerobot_path ../lerobot/src \
  --steps 500 \
  --agents 10
```
