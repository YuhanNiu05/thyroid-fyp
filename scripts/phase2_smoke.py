"""Small, non-research Qwen3-VL smoke test on the local TN5000 split.

Ground-truth labels are read only for the saved audit record and (in V1) the
assistant training target. They are never interpolated into an inference prompt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import re
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/official/TN5000_forReview"
OUT = ROOT / "outputs/phase2_smoke"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROMPT = (
    "Based only on this thyroid ultrasound image, classify the nodule. "
    "Reply with exactly one word: benign or malignant."
)


def package_versions() -> dict[str, str]:
    names = ["torch", "transformers", "accelerate", "qwen-vl-utils", "Pillow"]
    return {name: metadata.version(name) for name in names}


def parse_response(text: str) -> str | None:
    matches = re.findall(r"\b(benign|malignant)\b", text, flags=re.IGNORECASE)
    if not matches:
        return None
    normalized = {item.lower() for item in matches}
    return next(iter(normalized)) if len(normalized) == 1 else None


def make_messages(image_path: Path) -> list[dict]:
    # The filename contains only the image ID. No label, XML, ROI, or audit panel
    # is passed to the model.
    return [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": PROMPT},
    ]}]


def infer_one(model, processor, sample: dict, stage: str) -> dict:
    path = DATA / sample["image"]
    if sample["split"] != "val":
        raise ValueError(f"Inference sample is not from val: {sample['sample_id']}")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    record = {
        "stage": stage,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "ground_truth_for_audit_only": sample["class"],
        "prompt": PROMPT,
        "image_input": "original full image only",
    }
    try:
        messages = make_messages(path)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        answer = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        record.update({
            "raw_answer": answer,
            "parsed_prediction": parse_response(answer),
            "parse_status": "parsed" if parse_response(answer) else "parse_failed",
            "status": "success",
        })
    except Exception as exc:
        record.update({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
    record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    record["gpu_peak_allocated_mib"] = round(
        torch.cuda.max_memory_allocated() / (1024 ** 2), 2
    )
    record["gpu_allocated_before_mib"] = round(allocated_before / (1024 ** 2), 2)
    record["gpu_peak_delta_mib"] = round(
        max(torch.cuda.max_memory_allocated() - allocated_before, 0) / (1024 ** 2), 2
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["v0"], default="v0")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.limit <= 8:
        parser.error("--limit must be between 1 and 8")
    selection_path = OUT / "selection_seed20260923.json"
    selection = json.loads(selection_path.read_text())
    samples = selection["v0_val"][:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "v0_results.json"
    results = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "B / V0 zero-shot full-image inference",
        "model_id": MODEL_ID,
        "seed": selection["seed"],
        "selection_file": str(selection_path),
        "selected_samples": samples,
        "test_used": False,
        "prompt_ground_truth_free": True,
        "prompt": PROMPT,
        "packages": package_versions(),
        "gpu": torch.cuda.get_device_name(0),
        "load_status": "started",
        "results": [],
    }
    load_start = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype="auto", device_map="auto"
        )
        results["load_status"] = "success"
        results["model_load_seconds"] = round(time.perf_counter() - load_start, 3)
    except Exception as exc:
        results.update({
            "load_status": "failed",
            "model_load_seconds": round(time.perf_counter() - load_start, 3),
            "load_error": repr(exc),
            "load_traceback": traceback.format_exc(),
        })
        result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 2

    for sample in samples:
        record = infer_one(model, processor, sample, "v0")
        results["results"].append(record)
        result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        # Honor the required one-image gate before expanding to up to 8.
        if len(results["results"]) == 1 and record["status"] != "success":
            break
        if record["status"] != "success":
            break

    results["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    results["completed_count"] = len(results["results"])
    results["run_status"] = (
        "success" if len(results["results"]) == len(samples)
        and all(r["status"] == "success" for r in results["results"])
        else "partial_or_failed"
    )
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    return 0 if results["run_status"] == "success" else 3


if __name__ == "__main__":
    raise SystemExit(main())
