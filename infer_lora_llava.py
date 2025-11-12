import argparse
import os
from typing import Optional

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA + LoRA inference")
    parser.add_argument(r"--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt/question for the image",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=os.path.join("checkpoints", "final"),
        help="Path to the LoRA adapter folder (contains adapter_config.json)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="Base model to load (must match adapter base)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./models_cache",
        help="Local HF cache dir with model weights",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")
    return parser.parse_args()


def load_processor(base_model_id: str, cache_dir: Optional[str]) -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        use_fast=False,
    )
    return processor


def load_model_with_adapter(
    base_model_id: str,
    adapter_dir: str,
    cache_dir: Optional[str],
) -> torch.nn.Module:
    device_is_cuda = torch.cuda.is_available()
    dtype = torch.float16 if device_is_cuda else torch.float32

    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=None,  # Don't use auto device map to avoid offloading issues
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.eval()
    
    # Move model to appropriate device manually
    device = "cuda" if device_is_cuda else "cpu"
    model = model.to(device)
    
    return model


def build_prompt(processor: AutoProcessor, text_prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return prompt


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.isdir(args.adapter_dir):
        raise FileNotFoundError(f"Adapter dir not found: {args.adapter_dir}")

    processor = load_processor(args.base_model, args.cache_dir)
    model = load_model_with_adapter(args.base_model, args.adapter_dir, args.cache_dir)

    device_is_cuda = torch.cuda.is_available()
    device = "cuda" if device_is_cuda else "cpu"
    # Determine the model's floating dtype (e.g., fp16 on GPU)
    try:
        model_float_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_float_dtype = torch.float16 if device_is_cuda else torch.float32

    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(processor, args.prompt)

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    # Move tensors to device. Keep integer tensors as integer types; cast floating tensors to model dtype.
    prepared_inputs = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            prepared_inputs[key] = value
            continue
        if value.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            prepared_inputs[key] = value.to(device)
        else:
            prepared_inputs[key] = value.to(device=device, dtype=model_float_dtype)

    with torch.inference_mode():
        generated_ids = model.generate(
            **prepared_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("\n=== Model Output ===\n" + output_text)


if __name__ == "__main__":
    main()


