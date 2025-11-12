import argparse
import json
import os
from typing import List, Tuple

import torch
from PIL import Image
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from data import transform_all_vlm_to_llava, validate_llava_jsonl
from Train.MyTrainer import LlavaDataCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end pipeline: transform → train → infer")

    # Data
    parser.add_argument("--src_jsonl", type=str, required=True, help="Path to source JSONL (all_vlm_data_fixed.jsonl)")
    parser.add_argument("--images_root", type=str, default="", help="Images root for relative paths (optional)")
    parser.add_argument("--dataset_out_dir", type=str, default="json", help="Where to write train.jsonl/val.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="Fraction for validation split")

    # Model / cache
    parser.add_argument("--base_model", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--cache_dir", type=str, default="./models_cache")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated target modules",
    )

    # Train
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 if on CUDA")

    # Inference
    parser.add_argument("--infer_samples", type=int, default=3, help="How many val samples to run inference on")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")

    return parser.parse_args()


def split_llava_jsonl(src_path: str, out_dir: str, val_ratio: float) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")

    # Simple deterministic split (no shuffle to keep it reproducible offline)
    with open(src_path, "r", encoding="utf-8") as fin:
        lines = [ln for ln in fin if ln.strip()]

    n_total = len(lines)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    with open(train_path, "w", encoding="utf-8") as ftr:
        for ln in lines[:n_train]:
            ftr.write(ln if ln.endswith("\n") else ln + "\n")
    with open(val_path, "w", encoding="utf-8") as fva:
        for ln in lines[n_train:]:
            fva.write(ln if ln.endswith("\n") else ln + "\n")

    return train_path, val_path


def load_model_and_processor(base_model: str, cache_dir: str, fp16: bool) -> Tuple[AutoModelForVision2Seq, AutoProcessor]:
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if (use_cuda and fp16) else torch.float32

    processor = AutoProcessor.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model.config.use_cache = False
    return model, processor


def make_lora_model(model: AutoModelForVision2Seq, args: argparse.Namespace) -> AutoModelForVision2Seq:
    targets: List[str] = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
    )
    return get_peft_model(model, lora_cfg)


def preprocess_builder(processor: AutoProcessor):
    def _preprocess(ex):
        img_path = ex["image"]
        conv = ex.get("conversations", [])

        human_text = ""
        gpt_text = ""
        for turn in conv:
            if turn.get("from") == "human":
                human_text = turn.get("value", "")
            elif turn.get("from") == "gpt":
                gpt_text = turn.get("value", "")

        # Ensure single image placeholder in text
        image_token = getattr(processor, "image_token", None)
        placeholders = ["<image>"]
        if image_token and image_token not in placeholders:
            placeholders.append(image_token)
        for ph in placeholders:
            human_text = human_text.replace(ph + "\n", " ").replace(ph, " ")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": human_text},
                    {"type": "image"},
                ],
            }
        ]
        if gpt_text:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": gpt_text}]})

        # Build prompts for (a) user-only and (b) full user+assistant
        user_only_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": human_text},
                    {"type": "image"},
                ],
            }
        ]
        user_prompt = processor.apply_chat_template(
            user_only_messages, add_generation_prompt=False, tokenize=False
        )
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        # Token count for user segment; used by collator to mask labels
        user_ids = processor.tokenizer(
            user_prompt,
            add_special_tokens=True,
            return_attention_mask=False,
        ).input_ids
        user_len = len(user_ids)

        return {"prompt": prompt, "image": img_path, "user_len": user_len}

    return _preprocess


def generate_with_confidence(model, processor, image_path: str, prompt_text: str, args: argparse.Namespace) -> Tuple[str, float]:
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    try:
        model_float_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_float_dtype = torch.float16 if use_cuda and args.fp16 else torch.float32

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    prepared = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            prepared[k] = v
        elif v.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            prepared[k] = v.to(device)
        else:
            prepared[k] = v.to(device=device, dtype=model_float_dtype)

    with torch.inference_mode():
        out = model.generate(
            **prepared,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Decode text
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()

    # Confidence: average logprob of generated tokens
    scores = out.scores  # List[logits per step]
    ids = out.sequences[:, -len(scores) :]
    logprobs = []
    for step_logits, tok_id in zip(scores, ids[0]):
        lp = step_logits.log_softmax(-1)[0, tok_id.item()].item()
        logprobs.append(lp)
    avg_logprob = sum(logprobs) / max(1, len(logprobs))
    return text, avg_logprob


def main() -> None:
    args = parse_args()

    # 1) Transform → validate → split
    os.makedirs(args.dataset_out_dir, exist_ok=True)
    transformed_path = os.path.join(args.dataset_out_dir, "all_llava.jsonl")
    transform_all_vlm_to_llava(args.src_jsonl, transformed_path, images_root=args.images_root)
    report = validate_llava_jsonl(transformed_path, check_images_exist=False)
    print("[validate]", json.dumps(report, indent=2))

    train_path, val_path = split_llava_jsonl(transformed_path, args.dataset_out_dir, args.val_ratio)
    print(f"[split] train -> {train_path}\n[split] val   -> {val_path}")

    # 2) Load model + LoRA
    model, processor = load_model_and_processor(args.base_model, args.cache_dir, fp16=args.fp16)
    model = make_lora_model(model, args)

    # 3) Load dataset and preprocess
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    preprocess = preprocess_builder(processor)
    dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

    # 4) Trainer
    data_collator = LlavaDataCollator(tokenizer=processor.tokenizer, processor=processor)
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        fp16=bool(use_cuda and args.fp16),
        push_to_hub=False,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    print("[train] Starting training…")
    trainer.train()
    print("[train] Saving model to:", args.output_dir)
    trainer.save_model(args.output_dir)

    # 5) Inference + confidence on a few val samples
    print("[infer] Running quick inference on validation samples…")
    results = []
    # Load original val jsonl for raw prompts
    samples = []
    with open(val_path, "r", encoding="utf-8") as fval:
        for i, ln in enumerate(fval):
            if i >= args.infer_samples:
                break
            try:
                ex = json.loads(ln)
            except Exception:
                continue
            # Derive a prompt from the first human turn
            conv = ex.get("conversations", [])
            prompt = "Describe the image."
            for turn in conv:
                if turn.get("from") == "human":
                    prompt = str(turn.get("value", "")).replace("<image>", "").strip() or prompt
                    break
            samples.append((ex.get("image"), prompt))

    for img_path, prompt in samples:
        try:
            text, avg_logprob = generate_with_confidence(model, processor, img_path, prompt, args)
            results.append({
                "image": img_path,
                "prompt": prompt,
                "output": text,
                "avg_logprob": avg_logprob,
            })
        except Exception as e:
            results.append({"image": img_path, "prompt": prompt, "error": str(e)})

    out_json = os.path.join(args.output_dir, "inference_samples.json")
    with open(out_json, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f"[infer] Wrote {len(results)} samples -> {out_json}")


if __name__ == "__main__":
    main()


