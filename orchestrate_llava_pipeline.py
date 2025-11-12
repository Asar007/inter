import argparse
import json
import os
import subprocess
import sys
from typing import Tuple

from data import transform_all_vlm_to_llava, validate_llava_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orchestrate: transform → split → train (via existing llavaNextRun.py)")
    p.add_argument("--src_jsonl", required=True, help="Path to all_vlm_data_fixed.jsonl")
    p.add_argument("--images_root", default="", help="Optional images root for relative paths")
    p.add_argument("--dataset_out_dir", default="json", help="Output dir for train/val jsonl")
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--skip_train", action="store_true", help="Only prepare data; don't launch training")
    return p.parse_args()


def split_llava_jsonl(src_path: str, out_dir: str, val_ratio: float) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")

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


def main() -> None:
    args = parse_args()

    # 1) Transform
    os.makedirs(args.dataset_out_dir, exist_ok=True)
    transformed_path = os.path.join(args.dataset_out_dir, "all_llava.jsonl")
    transform_all_vlm_to_llava(args.src_jsonl, transformed_path, images_root=args.images_root)

    # 2) Validate
    report = validate_llava_jsonl(transformed_path, check_images_exist=False)
    print("[validate]", json.dumps(report))

    # 3) Split (deterministic)
    train_path, val_path = split_llava_jsonl(transformed_path, args.dataset_out_dir, args.val_ratio)
    print(f"[split] train -> {train_path}")
    print(f"[split] val   -> {val_path}")

    # 4) Train via existing llavaNextRun.py (it reads from json/train.jsonl & json/val.jsonl)
    if not args.skip_train:
        print("[train] Launching training using llavaNextRun.py …")
        cmd = [sys.executable, "llavaNextRun.py"]
        # Inherit environment and working directory
        subprocess.run(cmd, check=True)
        print("[train] Finished.")

    print("Done. Use infer_lora_llava.py for inference.")


if __name__ == "__main__":
    main()


