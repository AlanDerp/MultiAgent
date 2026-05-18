"""
pi0_base 推理测试脚本（无需真实机器人硬件）
修复了以下问题：
  1. image key 名与模型 config 匹配
  2. 语言输入改为预 tokenize 方式
  3. sentencepiece tokenizer 正确加载

依赖安装：
    pip install -e ".[pi]"
    pip install sentencepiece
"""

import sys
import traceback
import torch

# ─────────────────────────────────────────────
def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")

def ok(msg):   print(f"  ✅ {msg}")
def fail(msg): print(f"  ❌ {msg}")
def info(msg): print(f"  ℹ️  {msg}")


# ─────────────────────────────────────────────
# STEP 1：环境检查
# ─────────────────────────────────────────────
section("STEP 1 · 环境检查")

info(f"Python {sys.version.split()[0]}")
info(f"PyTorch {torch.__version__}")

if torch.cuda.is_available():
    ok(f"CUDA 可用：{torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    ok("MPS (Apple Silicon) 可用")
    DEVICE = "mps"
else:
    info("仅 CPU 可用")
    DEVICE = "cpu"

device = torch.device(DEVICE)

try:
    import lerobot
    ok(f"lerobot {lerobot.__version__}")
except ImportError:
    fail("lerobot 未安装，请运行：pip install -e \".[pi]\"")
    sys.exit(1)

try:
    import sentencepiece
    ok(f"sentencepiece {sentencepiece.__version__}")
except ImportError:
    fail("sentencepiece 未安装，请运行：pip install sentencepiece")
    sys.exit(1)


# ─────────────────────────────────────────────
# STEP 2：加载 pi0_base 策略
# ─────────────────────────────────────────────
section("STEP 2 · 加载 pi0_base 策略")

MODEL_ID = "lerobot/pi0_base"

try:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    info(f"正在加载模型：{MODEL_ID}（首次运行需要下载，约 8 GB）")
    policy = PI0Policy.from_pretrained(MODEL_ID)
    policy = policy.to(device).eval()
    ok(f"模型加载成功，运行于 {DEVICE.upper()}")

except Exception as e:
    fail(f"模型加载失败：{e}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────────────────────────────────────
# STEP 2.5：打印模型期望的输入 key（调试用）
# ─────────────────────────────────────────────
section("STEP 2.5 · 模型期望的输入规格")

info(f"图像 features：{list(policy.config.image_features.keys())}")
info(f"所有输入 features：{list(policy.config.input_features.keys())}")
info(f"输出 features：{list(policy.config.output_features.keys())}")
info(f"tokenizer_max_length：{policy.config.tokenizer_max_length}")

IMAGE_KEYS = list(policy.config.image_features.keys())
STATE_DIM = policy.config.max_state_dim
TOKEN_LEN = policy.config.tokenizer_max_length


# ─────────────────────────────────────────────
# STEP 3：加载 Tokenizer
# ─────────────────────────────────────────────
section("STEP 3 · 加载语言 Tokenizer")

tokenizer = None

# 尝试1：直接从模型 hub 加载
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ok(f"Tokenizer 加载成功（来自 {MODEL_ID}）：{type(tokenizer).__name__}")
except Exception as e1:
    info(f"从 {MODEL_ID} 加载失败：{e1}")

# 尝试2：从 PaliGemma 加载（pi0 底层用的是 PaliGemma）
if tokenizer is None:
    try:
        from transformers import AutoTokenizer
        info("尝试从 google/paligemma-3b-pt-224 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        ok(f"PaliGemma Tokenizer 加载成功：{type(tokenizer).__name__}")
    except Exception as e2:
        fail(f"PaliGemma Tokenizer 加载失败：{e2}")
        info("请确保：")
        info("  1. 已申请 PaliGemma 访问权限：https://huggingface.co/google/paligemma-3b-pt-224")
        info("  2. 已登录 HuggingFace：huggingface-cli login")
        traceback.print_exc()
        sys.exit(1)


# ─────────────────────────────────────────────
# STEP 4：Tokenize 任务描述
# ─────────────────────────────────────────────
section("STEP 4 · Tokenize 任务描述")

TASK_TEXT = "Place the pink block in the bowl"

try:
    encoded = tokenizer(
        TASK_TEXT,
        return_tensors="pt",
        padding="max_length",
        max_length=TOKEN_LEN,
        truncation=True,
    )

    lang_tokens = encoded["input_ids"].to(device)        # (1, TOKEN_LEN)
    lang_mask   = encoded["attention_mask"].bool().to(device)   # (1, TOKEN_LEN)

    ok(f"lang_tokens shape：{lang_tokens.shape}")
    ok(f"lang_mask   shape：{lang_mask.shape}")

except Exception as e:
    fail(f"Tokenize 失败：{e}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────────────────────────────────────
# STEP 5：构造 batch 并推理
# ─────────────────────────────────────────────
section("STEP 5 · Mock 推理测试")

try:
    batch = {}

    # 图像输入（动态读取模型期望的 key，不硬编码）
    for img_key in IMAGE_KEYS:
        batch[img_key] = torch.rand(1, 3, 224, 224, device=device)
        info(f"图像 key：{img_key}  shape：{batch[img_key].shape}")

    # 状态输入
    batch["observation.state"] = torch.randn(1, STATE_DIM, device=device)
    info(f"state shape：{batch['observation.state'].shape}")

    # 语言 token（pi0_base 要求预 tokenize，不接受原始字符串）
    batch["observation.language.tokens"]         = lang_tokens
    batch["observation.language.attention_mask"] = lang_mask

    with torch.no_grad():
        action = policy.select_action(batch)

    ok(f"推理成功！输出动作 shape：{action.shape}")
    info(f"动作数值范围：[{action.min():.4f}, {action.max():.4f}]")
    info(f"动作前5维：{action[0, :5].cpu().numpy()}")

except Exception as e:
    fail(f"推理失败：{e}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────────────────────────────────────
# STEP 6：多步动作序列
# ─────────────────────────────────────────────
section("STEP 6 · 批量动作序列生成")

try:
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)
    ok(f"动作序列 shape：{chunk.shape}  "
       f"(batch={chunk.shape[0]}, steps={chunk.shape[1]}, dim={chunk.shape[2]})")
except Exception as e:
    fail(f"批量生成失败：{e}")
    traceback.print_exc()


# ─────────────────────────────────────────────
section("🎉 全部测试完成！")
print("""
  后续步骤：
  1. 对接真实摄像头  → 替换 torch.rand(...) 为 OpenCV/RealSense 帧
  2. 对接机器臂      → 将 action 输出送到关节控制器
  3. 微调            → 运行 lerobot-train（见 finetune_pi0.sh）
""")