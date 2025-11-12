from torch.utils.data import DataLoader
from data import get_dataloader
from tqdm import tqdm
import os, json, torch, sys
from transformers import AutoProcessor, AutoModelForVision2Seq
try:
    from transformers import BitsAndBytesConfig  # optional, for 4-bit quant
except Exception:
    BitsAndBytesConfig = None
import peft
print(peft.__version__)
from peft import get_peft_model, LoraConfig, TaskType
try:
    from peft import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None
from datasets import load_dataset
from PIL import Image
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from Train.MyTrainer import run_training

# =====================
# CONFIGURATION
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

JSONL_PATH = r"E:\vlm\json"  # Folder containing train.jsonl and val.jsonl
IMAGES_FOLDER = r"E:\vlm\images"
OUTPUT_PATH = r"E:\vlm\predictions.jsonl"

print(f"✅ Using device: {DEVICE}")
print(list(TaskType))

# =====================
# MODEL & PROCESSOR
# =====================
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
processor_name = MODEL_ID
CACHE_DIR = os.getenv("HF_HOME", r"./models_cache")

# Load processor
processor = AutoProcessor.from_pretrained(
    processor_name,
    cache_dir=CACHE_DIR,
    use_fast=True,
    local_files_only=True
)
print(f"✅ Processor loaded from: {processor_name}")

use_cuda = torch.cuda.is_available()
quant_config = None
if BitsAndBytesConfig is not None:
    try:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if use_cuda else torch.float32,
        )
        print("✅ Using 4-bit quantization (QLoRA) via bitsandbytes")
    except Exception:
        quant_config = None

base_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    device_map="auto" if use_cuda else None,
    low_cpu_mem_usage=True,
    cache_dir=CACHE_DIR,
    local_files_only=True,
    quantization_config=quant_config,
)
# Do not force-move entire model to a single device when using device_map/quant
base_model.config.use_cache = False
if quant_config is not None and prepare_model_for_kbit_training is not None:
    try:
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
        print("✅ Prepared model for k-bit training")
    except Exception:
        print("⚠️ Could not prepare model for k-bit training; continuing.")
print(f"✅ Base model loaded from: {MODEL_ID}")

print("🔄 Setting up LoRA configuration...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adjust based on inspection
    lora_dropout=0.1
)
model = get_peft_model(base_model, lora_config)
if hasattr(model, "enable_input_require_grads"):
    try:
        model.enable_input_require_grads()
        print("✅ Enabled input require grads")
    except Exception:
        pass

# =====================
# DATA LOADING
# =====================
dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(JSONL_PATH, "train.jsonl"),
        "validation": os.path.join(JSONL_PATH, "val.jsonl")
    }
)

# =====================
# PREPROCESS FUNCTION
# =====================
def preprocess(ex):
    img_path = ex["image"]
    print(f"Image path: {img_path}")

    # Build chat conversation using the expected template with an image placeholder
    conv = ex.get("conversations", [])
    human_text, gpt_text = "", ""
    for turn in conv:
        if turn.get("from") == "human":
            human_text = turn.get("value", "")
        elif turn.get("from") == "gpt":
            gpt_text = turn.get("value", "")

    # Remove any pre-inserted image placeholder from the dataset text to avoid double <image> tokens
    image_token = getattr(processor, "image_token", None)
    placeholders = ["<image>"]
    if image_token and image_token not in placeholders:
        placeholders.append(image_token)
    for ph in placeholders:
        if human_text:
            human_text = human_text.replace(ph + "\n", " ")
            human_text = human_text.replace(ph, " ")

    messages = []
    # Always include an image placeholder in the user turn to align with images input
    user_text = human_text or ""
    # Put image first, then text, to match LLaVA-Next expected ordering
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ],
    })
    if gpt_text:
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": gpt_text},
            ],
        })

    # For supervised fine-tuning include assistant text in the prompt (no generation prompt)
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    # Also build the user-only prompt to compute token length for masking in the collator
    user_only_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    user_prompt = processor.apply_chat_template(
        user_only_messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    user_ids = processor.tokenizer(
        user_prompt,
        add_special_tokens=True,
        return_attention_mask=False,
    ).input_ids
    user_len = len(user_ids)

    # Do not preprocess here; return prompt and image path so the collator can batch-process with padding
    return {"prompt": prompt, "image": img_path, "user_len": user_len}

dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# =====================
# TRAINING
# =====================
print("🔄 Running training...")
run_training(model, dataset["train"], dataset["validation"], processor)
