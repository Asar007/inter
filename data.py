# data.py
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

class VLMDataset(Dataset): # <--- THIS ENTIRE CLASS DEFINITION MUST BE HERE
    def __init__(self, jsonl_path, images_folder):
        self.images_folder = images_folder
        self.data = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        if not os.path.exists(images_folder):
            raise FileNotFoundError(f"Images folder not found: {images_folder}")

        skipped = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    skipped += 1
                    continue
                try:
                    entry = json.loads(line)
                    if "image" not in entry:
                        print(f"Skipping line {i} due to missing 'image' key: {line[:100]}...")
                        skipped += 1
                        continue

                    if "text" not in entry and "blip_caption" not in entry:
                        print(f"Skipping line {i} due to missing 'text' AND 'blip_caption' key: {line[:100]}...")
                        skipped += 1
                        continue

                    self.data.append(entry)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line {i}: {line[:100]}...")
                    skipped += 1
        print(f"Loaded {len(self.data)} entries. Skipped {skipped} lines.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # This part depends on your JSONL. Choose one:
        # If your JSONL contains absolute paths like "C:\\Users\\...", use:
        img_path = entry["image"]
        # If your JSONL contains relative paths like "image.jpg" and IMAGES_FOLDER is the base, use:
        # img_path = os.path.join(self.images_folder, entry["image"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning a placeholder.")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        text = entry.get("text", entry.get("blip_caption", ""))
        
        return {"image": image, "text": text, "id": entry.get("id", str(idx))}

# Define a custom collate function for the DataLoader
def custom_collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    ids = [item['id'] for item in batch] 

    return {
        "image": images,
        "text": texts,
        "id": ids
    }

def get_dataloader(jsonl_path, images_folder, batch_size=4, shuffle=False):
    dataset = VLMDataset(jsonl_path, images_folder) # <--- VLMDataset is used here
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Dataset sample: {dataset[0]}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
# ----------------------
# LLaVA JSONL Transformer
# ----------------------
def transform_all_vlm_to_llava(
    input_jsonl_path: str,
    output_jsonl_path: str,
    images_root: str = "",
    prefer_annotations: bool = True,
    default_instruction: str = "Describe the object and activity in this image.",
) -> None:
    """
    Transform a general JSONL dataset (with keys like 'blip_caption' and 'annotations'
    containing 'instruction' and 'answer') into the LLaVA conversations format:

    {"image": "<path>", "conversations": [
        {"from": "human", "value": "<image>\n<instruction>"},
        {"from": "gpt",   "value": "<answer>"}
    ]}

    - images_root: prepended if image paths are relative; leave empty to keep absolute paths.
    - prefer_annotations: use first annotation's instruction/answer if present; otherwise fall back to blip_caption.
    - default_instruction: used when no instruction is available.
    """
    total = 0
    written = 0
    skipped = 0

    with open(input_jsonl_path, "r", encoding="utf-8") as fin, open(output_jsonl_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                print(f"[transform] Skipping invalid JSON at line {line_num}")
                skipped += 1
                continue

            image_path = ex.get("image")
            if not image_path:
                print(f"[transform] Missing 'image' at line {line_num}")
                skipped += 1
                continue

            # Normalize image path
            if images_root and not os.path.isabs(image_path):
                image_path = os.path.join(images_root, image_path)

            instruction: str = ""
            answer: str = ""

            if prefer_annotations:
                annotations = ex.get("annotations")
                if isinstance(annotations, list) and len(annotations) > 0:
                    first = annotations[0] or {}
                    instruction = str(first.get("instruction", "")).strip()
                    answer = str(first.get("answer", "")).strip()

            # Fallbacks
            if not instruction:
                instruction = ex.get("instruction") or ex.get("prompt") or default_instruction
                instruction = str(instruction).strip()

            if not answer:
                answer = ex.get("text") or ex.get("caption") or ex.get("blip_caption") or ""
                answer = str(answer).strip()

            if not answer:
                print(f"[transform] Missing answer/caption at line {line_num}")
                skipped += 1
                continue

            conversations = [
                {"from": "human", "value": f"<image>\n{instruction}"},
                {"from": "gpt", "value": answer},
            ]

            out = {"image": image_path, "conversations": conversations}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"[transform] Done. Total: {total}, Written: {written}, Skipped: {skipped}. Output -> {output_jsonl_path}")


# ----------------------
# LLaVA JSONL Validator
# ----------------------
def validate_llava_jsonl(
    jsonl_path: str,
    check_images_exist: bool = True,
    sample_limit: int = 0,
) -> dict:
    """
    Validate a LLaVA conversations JSONL file. Returns a summary dict with counts.

    Checks:
    - valid JSON per line
    - presence of 'image' and 'conversations'
    - conversations is a non-empty list with required 'from' and 'value' keys
    - first user message contains an image placeholder
    - optional: image file existence
    """
    summary = {
        "total": 0,
        "valid": 0,
        "invalid_json": 0,
        "missing_image": 0,
        "missing_conversations": 0,
        "bad_conversation_shape": 0,
        "no_image_placeholder": 0,
        "missing_image_file": 0,
    }

    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            summary["total"] += 1
            if sample_limit and summary["total"] > sample_limit:
                break

            line = line.strip()
            if not line:
                summary["invalid_json"] += 1
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                summary["invalid_json"] += 1
                continue

            image_path = ex.get("image")
            conversations = ex.get("conversations")

            if not image_path:
                summary["missing_image"] += 1
                continue
            if not isinstance(conversations, list) or len(conversations) == 0:
                summary["missing_conversations"] += 1
                continue

            ok_shape = True
            for msg in conversations:
                if not isinstance(msg, dict) or "from" not in msg or "value" not in msg:
                    ok_shape = False
                    break
            if not ok_shape:
                summary["bad_conversation_shape"] += 1
                continue

            first = conversations[0]
            if first.get("from") != "human" or "<image>" not in str(first.get("value", "")):
                summary["no_image_placeholder"] += 1
                continue

            if check_images_exist and not os.path.exists(image_path):
                summary["missing_image_file"] += 1
                continue

            summary["valid"] += 1

    return summary


if __name__ == "__main__":
    # Example CLI usage for quick transforms/validation
    src = os.environ.get("SRC_JSONL", "E:/vlm/json/all_vlm_data_fixed.jsonl")
    dst = os.environ.get("DST_JSONL", "E:/vlm/json/train.jsonl")
    imgs = os.environ.get("IMAGES_ROOT", "")

    transform_all_vlm_to_llava(src, dst, images_root=imgs)
    report = validate_llava_jsonl(dst, check_images_exist=False, sample_limit=0)
    print("[validate]", report)
