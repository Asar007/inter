from transformers import Trainer, TrainingArguments
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

class MyTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        # Skip moving model to device since it's already managed by device_map
        return model


@dataclass
class LlavaDataCollator:
    tokenizer: Any
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Convert prompts and images to model features in one call so processor can align image tokens
        texts = [f["prompt"] for f in features]
        images: List[Any] = []
        image_paths: List[Any] = []
        for f in features:
            img = f["image"]
            image_paths.append(img)
            if isinstance(img, str):
                from PIL import Image
                img = Image.open(img).convert("RGB")
            images.append(img)

        # Important: do NOT truncate here for LLaVA-Next. Truncation can cut image tokens
        # and cause mismatch between image features and image token count.
        # Let the processor handle sequence construction and padding.
        proc_out = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        # Labels construction with configurable modes
        input_ids = proc_out["input_ids"]
        attention_mask = proc_out.get("attention_mask")
        labels = input_ids.clone()

        label_mode = os.getenv("LLAVA_LABEL_MODE", "assistant_only").strip().lower()

        # Helper: infer class label from feature or image path
        default_classes = ["wiping", "swipes", "droplet", "pool", "pattern", "bloodflow"]
        classes_env = os.getenv("LLAVA_CLASSES", ",".join(default_classes))
        class_vocab = [c.strip().lower() for c in classes_env.split(",") if c.strip()]

        def infer_class_label(feature: Dict[str, Any], path_like: Any) -> Any:
            label = feature.get("class_label")
            if isinstance(label, str) and label:
                return label
            if isinstance(path_like, str):
                lower_path = path_like.lower()
                for cls in class_vocab:
                    if cls in lower_path:
                        return cls
            return None

        if label_mode == "class_only":
            # Mask all by default; then unmask and set targets to final class token(s)
            labels.fill_(-100)
            for i, (feat, path_like) in enumerate(zip(features, image_paths)):
                target_text = infer_class_label(feat, path_like)
                if not target_text:
                    # Fallback: assistant-only masking when class cannot be determined
                    user_len = int(feat.get("user_len", 0) or 0)
                    if user_len > 0:
                        labels[i, : user_len] = -100
                    if attention_mask is not None:
                        labels[i, attention_mask[i] == 0] = -100
                    continue
                target_ids = self.tokenizer(
                    target_text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                ).input_ids
                k = len(target_ids)
                if k == 0:
                    continue
                seq_len = input_ids.shape[1]
                start_pos = max(0, seq_len - k)
                labels[i, start_pos:seq_len] = torch.tensor(target_ids[: seq_len - start_pos])
                if attention_mask is not None:
                    labels[i, attention_mask[i] == 0] = -100
        else:
            # assistant_only (default): mask user/prompt tokens using user_len and mask padding
            for i, feat in enumerate(features):
                user_len = int(feat.get("user_len", 0) or 0)
                if user_len > 0:
                    labels[i, : user_len] = -100
                if attention_mask is not None:
                    labels[i, attention_mask[i] == 0] = -100

        proc_out["labels"] = labels
        return proc_out


def run_training(model, train_dataset, eval_dataset, processor):
    # Allow tuning for memory constraints via env vars
    per_device_train_batch_size = int(os.getenv("LLAVA_TRAIN_BATCH", "1"))
    per_device_eval_batch_size = int(os.getenv("LLAVA_EVAL_BATCH", "1"))
    grad_accum = int(os.getenv("LLAVA_GRAD_ACCUM", "4"))
    num_epochs = int(os.getenv("LLAVA_EPOCHS", "1"))
    use_fp16 = os.getenv("LLAVA_FP16", "1") == "1"
    learning_rate = float(os.getenv("LLAVA_LR", "1e-4"))
    warmup_ratio = float(os.getenv("LLAVA_WARMUP", "0.03"))

    training_args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=use_fp16,
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    data_collator = LlavaDataCollator(tokenizer=processor.tokenizer, processor=processor)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("checkpoints/final")
    print("✅ Training complete. Model saved to checkpoints/final.")
