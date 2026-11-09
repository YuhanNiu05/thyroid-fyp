"""Compare the saved whole-image model with an independent, single-image ROI view.

The ground-truth class is used only for audit and scoring. Images and crops stay
under the ignored local data/output directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/official/TN5000_forReview/JPEGImages"
HISTORY = ROOT / "outputs/final_val496"
MODEL = (
    ROOT
    / "outputs/model_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct"
    / "snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
ADAPTER = ROOT / "outputs/checkpoint-epoch-2"
OUTPUT = ROOT / "outputs/phase1_roi_only"
PROMPT = (
    "Based only on this thyroid ultrasound image, classify the nodule. "
    "Reply with exactly one word: benign or malignant."
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_reference() -> tuple[dict[str, dict], dict[str, dict]]:
    with (HISTORY / "per_case_results.csv").open(newline="") as handle:
        whole = {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["arm"] == "V1"
        }
    roi = {
        row["sample_id"]: row
        for row in json.loads((HISTORY / "o1_val_results.json").read_text())["records"]
    }
    assert len(whole) == len(roi) == 496 and set(whole) == set(roi)
    assert sum(row["true_class"] == "benign" for row in whole.values()) == 123
    assert all(
        whole[sid]["true_class"] == roi[sid]["true_class_for_metrics_only"]
        for sid in whole
    )
    return whole, roi


def make_inputs(processor, image: Image.Image) -> tuple[dict, dict]:
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(messages, image_patch_size=14)
    assert videos is None and images is not None and len(images) == 1
    encoded = processor(
        text=[rendered], images=images, padding=True, return_tensors="pt", truncation=False
    )
    grid = encoded["image_grid_thw"].tolist()
    tokens = grid[0][0] * grid[0][1] * grid[0][2] // 4
    details = {
        "preprocessor_size_wh": list(images[0].size),
        "processor_size_wh": [grid[0][2] * 16, grid[0][1] * 16],
        "vision_grid_thw": grid[0],
        "vision_tokens": tokens,
        "input_tokens": int(encoded["input_ids"].shape[1]),
    }
    return encoded, details


def infer(model, processor, image: Image.Image) -> dict:
    start = time.perf_counter()
    encoded, details = make_inputs(processor, image)
    encoded = encoded.to(model.device)
    with torch.inference_mode():
        generated = model.generate(**encoded, do_sample=False, max_new_tokens=24)
    torch.cuda.synchronize()
    answer_ids = generated[0, encoded["input_ids"].shape[1] :]
    raw = processor.decode(answer_ids, skip_special_tokens=True).strip()
    prediction = raw.lower() if raw.lower() in {"benign", "malignant"} else None
    details.update(
        raw_answer=raw,
        prediction=prediction,
        generated_tokens=len(answer_ids),
        elapsed_seconds=time.perf_counter() - start,
    )
    return details


def validate_view(sid: str, view: str, result: dict, whole: dict, roi: dict) -> None:
    if view == "whole":
        old = whole[sid]
        if result["raw_answer"] != old["raw_answer"]:
            raise RuntimeError(
                f"V1 reproduction failed for {sid}: {result['raw_answer']!r} "
                f"vs {old['raw_answer']!r}"
            )
        if result["vision_tokens"] != int(old["total_visual_tokens"]):
            raise RuntimeError(f"V1 visual-token mismatch for {sid}")
    elif result["prediction"] is None:
        # Keep invalid answers in the output; the report counts them as errors.
        print(f"ROI parse failure: {sid}: {result['raw_answer']!r}", flush=True)
    if not result["vision_tokens"] or result["generated_tokens"] >= 24:
        raise RuntimeError(f"Possible truncation or empty visual input: {sid} {view}")


def case_images(sid: str, saved: dict) -> tuple[Image.Image, Image.Image, list[int]]:
    with Image.open(DATA / f"{sid}.jpg") as source:
        full = source.convert("RGB")
    box = saved["crop_xyxy_half_open"]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= full.width and 0 <= y1 < y2 <= full.height):
        raise RuntimeError(f"Bad ROI coordinates: {sid}: {box}")
    cropped = full.crop(tuple(box))
    digest = sha256(str(cropped.size).encode() + b"\0" + cropped.tobytes())
    if digest != saved["roi_pixel_sha256"]:
        raise RuntimeError(f"ROI pixel mismatch: {sid}")
    return full, cropped, box


def run_case(sid: str, model, processor, whole: dict, roi: dict) -> dict:
    full, cropped, box = case_images(sid, roi[sid])
    w = infer(model, processor, full)
    validate_view(sid, "whole", w, whole, roi)
    r = infer(model, processor, cropped)
    validate_view(sid, "roi", r, whole, roi)
    if w["processor_size_wh"] != json.loads(whole[sid]["actual_sizes_wh"])[0]:
        raise RuntimeError(f"V1 preprocessor-size mismatch for {sid}")
    if r["preprocessor_size_wh"] != roi[sid]["qwen_vl_utils_size_wh"][1]:
        raise RuntimeError(f"ROI preprocessor-size mismatch for {sid}")
    if r["processor_size_wh"] != roi[sid]["actual_processor_size_wh"][1]:
        raise RuntimeError(f"ROI model-size mismatch for {sid}")
    return {
        "sample_id": sid,
        "true_class": whole[sid]["true_class"],
        "full_size_wh": list(full.size),
        "roi_size_wh": list(cropped.size),
        "roi_box_xyxy": box,
        "whole": w,
        "roi": r,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT / "predictions.jsonl")
    args = parser.parse_args()
    whole, roi = load_reference()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this saved checkpoint")
    expected_adapter = (
        "e7541083da9725aa988408e431497007ce4e438117f8708c42d44a0f3a38b71c"
    )
    adapter_hash = sha256((ADAPTER / "adapter_model.safetensors").read_bytes())
    if adapter_hash != expected_adapter:
        raise RuntimeError("Adapter SHA256 does not match saved experiment")
    torch.manual_seed(20260923)
    torch.backends.cuda.matmul.allow_tf32 = True
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    )
    model = PeftModel.from_pretrained(base, ADAPTER, local_files_only=True).eval()

    gate = []
    for label in ("benign", "malignant"):
        gate.extend(
            sorted(sid for sid, row in whole.items() if row["true_class"] == label)[:3]
        )
    for sid in gate:
        run_case(sid, model, processor, whole, roi)
    print(f"Gate passed: {len(gate)} cases", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w") as handle:
        for sid in sorted(whole):
            record = run_case(sid, model, processor, whole, roi)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            if count % 50 == 0 or count == 496:
                print(f"Completed {count}/496", flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
