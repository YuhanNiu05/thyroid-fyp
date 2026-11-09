"""Train ROI-only (V2) or whole-image plus ROI (V3) Qwen3-VL LoRA models.

This script deliberately reuses the frozen V1 split construction, optimizer,
LoRA, batching, shuffling, validation, parsing, and checkpoint-selection rules.
The only experimental difference between V2 and V3 is the visual input and its
ground-truth-free prompt. Test images are never loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
import random
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import phase2_train256 as v1_development
import phase2_train_full as v1_full
from phase2_lora_smoke import versions
from phase2_smoke import DATA, MODEL_ID, ROOT, parse_response
from phase2_train256 import metrics, sha256_file, write_json


SEED = 20260923
FULL_EPOCHS = 5
EFFECTIVE_BATCH_SIZE = 2
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EXPECTED_MANIFEST_SHA256 = (
    "e2cdf2ad6db1591114db813169d616183c313ec1581c735d1bd9dca1f6f2fa2c"
)
EXPECTED_SELECTION_SHA256 = (
    "ae8ce2d31482594cde96c2764a90aad4fe0f7f8bd0facd79c6f391753b81cec7"
)
EXPECTED_V1_ADAPTER_SHA256 = (
    "e7541083da9725aa988408e431497007ce4e438117f8708c42d44a0f3a38b71c"
)
MODEL_CACHE = ROOT / "outputs/model_cache"
MODEL_HUB_CACHE = MODEL_CACHE / "hub"
RUN_ROOT = ROOT / "outputs"
V1_RESULTS = ROOT / "outputs/final_val496/per_case_results.csv"
V1_ADAPTER = ROOT / "outputs/checkpoint-epoch-2/adapter_model.safetensors"
MANIFEST_CANDIDATES = [
    ROOT / "reports/tn5000_seed20260923_final/manifest.jsonl",
    ROOT
    / ".archive/legacy_20261007/reports/tn5000_seed20260923_final/manifest.jsonl",
]

PROMPTS = {
    "v2": (
        "Based only on this manual ROI crop of a thyroid ultrasound nodule, "
        "classify the nodule. Reply with exactly one word: benign or malignant."
    ),
    "v3": (
        "Based only on these thyroid ultrasound images, classify the nodule. "
        "The first image is the full ultrasound image; the second image is a "
        "manual ROI crop of the same nodule. Reply with exactly one word: benign "
        "or malignant."
    ),
}

INPUT_DESCRIPTIONS = {
    "v2": "manual annotation ROI crop only",
    "v3": "image 1 original whole image; image 2 matching manual ROI crop",
}


def emit(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a") as handle:
        handle.write(message + "\n")
        handle.flush()


def json_sha256(value: object) -> str:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    return hashlib.sha256(data).hexdigest()


def resolve_manifest() -> Path:
    for path in MANIFEST_CANDIDATES:
        if path.is_file():
            if sha256_file(path) != EXPECTED_MANIFEST_SHA256:
                raise RuntimeError(f"Manifest SHA256 mismatch: {path}")
            return path
    raise FileNotFoundError(f"No frozen manifest found: {MANIFEST_CANDIDATES}")


def rebuild_frozen_v1_selection(manifest: Path) -> dict:
    """Rebuild the exact V1 selection after its old run directory was archived."""
    v1_development.MANIFEST = manifest
    prior_selection = v1_development.build_selection()
    with tempfile.TemporaryDirectory(prefix="tn5000_v1_selection_") as directory:
        prior_dir = Path(directory)
        write_json(prior_dir / "selection.json", prior_selection)
        v1_full.MANIFEST = manifest
        v1_full.PRIOR_RUN = prior_dir
        selection = v1_full.build_full_selection()
    actual_hash = json_sha256(selection)
    if actual_hash != EXPECTED_SELECTION_SHA256:
        raise RuntimeError(
            f"Frozen V1 selection mismatch: {actual_hash} != {EXPECTED_SELECTION_SHA256}"
        )
    with V1_RESULTS.open(newline="") as handle:
        saved_ids = [
            row["sample_id"] for row in csv.DictReader(handle) if row["arm"] == "V1"
        ]
    current_ids = [sample["sample_id"] for sample in selection["validation_samples"]]
    if saved_ids != current_ids:
        raise RuntimeError("Rebuilt validation order does not exactly match V1 val496")
    return selection


def load_roi_manifest(manifest: Path, selection: dict) -> dict:
    by_id = {
        record["sample_id"]: record
        for record in (json.loads(line) for line in manifest.read_text().splitlines())
    }
    entries = {}
    selected = selection["train_samples"] + selection["validation_samples"]
    for sample in selected:
        record = by_id[sample["sample_id"]]
        if not record["structurally_valid"] or len(record["objects"]) != 1:
            raise RuntimeError(f"ROI input is not structurally valid: {sample['sample_id']}")
        obj = record["objects"][0]
        box = [int(value) for value in obj["crop_xyxy_half_open"]]
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= record["width"] and 0 <= y1 < y2 <= record["height"]):
            raise RuntimeError(f"Invalid ROI box for {sample['sample_id']}: {box}")
        entries[sample["sample_id"]] = {
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "annotation": sample["annotation"],
            "crop_xyxy_half_open": box,
            "roi_size_wh": [x2 - x1, y2 - y1],
            "coordinate_rule": (
                "official loader coordinates: xmin,ymin,xmax,ymax each minus 1; "
                "Pillow half-open crop; no padding or clamping"
            ),
        }
    if len(entries) != 3880:
        raise RuntimeError(f"Expected 3880 train+val ROI entries, got {len(entries)}")
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "entry_count": len(entries),
        "coordinate_rule": (
            "All four XML coordinates minus 1, then Pillow half-open crop; "
            "no padding, expansion, clamping, or source-file modification."
        ),
        "entries": entries,
    }


def load_views(sample: dict, mode: str, roi_entries: dict) -> list[Image.Image]:
    with Image.open(DATA / sample["image"]) as source:
        whole = source.convert("RGB")
    box = roi_entries[sample["sample_id"]]["crop_xyxy_half_open"]
    roi = whole.crop(tuple(box))
    expected_size = tuple(roi_entries[sample["sample_id"]]["roi_size_wh"])
    if roi.size != expected_size:
        raise RuntimeError(
            f"ROI size mismatch for {sample['sample_id']}: {roi.size} != {expected_size}"
        )
    if mode == "v2":
        return [roi]
    if mode == "v3":
        return [whole, roi]
    raise ValueError(f"Unknown mode: {mode}")


def make_messages(sample: dict, mode: str, roi_entries: dict) -> list[dict]:
    images = load_views(sample, mode, roi_entries)
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": PROMPTS[mode]})
    return [{"role": "user", "content": content}]


def input_pair(processor, sample: dict, mode: str, roi_entries: dict):
    user_messages = make_messages(sample, mode, roi_entries)
    full_messages = user_messages + [
        {"role": "assistant", "content": sample["class"]}
    ]
    user_text = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(user_messages)
    expected_images = 1 if mode == "v2" else 2
    if video_inputs is not None or image_inputs is None or len(image_inputs) != expected_images:
        raise RuntimeError(
            f"Unexpected visual input count for {sample['sample_id']}: "
            f"{None if image_inputs is None else len(image_inputs)}"
        )
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        truncation=False,
    )
    user_inputs = processor(
        text=[user_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        truncation=False,
    )
    prefix_len = user_inputs["input_ids"].shape[-1]
    if not torch.equal(full_inputs["input_ids"][:, :prefix_len], user_inputs["input_ids"]):
        raise RuntimeError("Prompt tokens are not a prefix of the supervised conversation")
    labels = full_inputs["input_ids"].clone()
    labels[:, :prefix_len] = -100
    supervised = labels[labels != -100]
    if supervised.numel() == 0:
        raise RuntimeError("No assistant target tokens remain after prompt masking")
    return full_inputs, labels, {
        "input_image_count": expected_images,
        "prompt_token_count": int(prefix_len),
        "supervised_token_count": int(supervised.numel()),
        "vision_grid_thw": full_inputs["image_grid_thw"].tolist(),
    }


def infer_one(
    model,
    processor,
    sample: dict,
    mode: str,
    roi_entries: dict,
    stage: str,
) -> dict:
    if sample["split"] != "val":
        raise ValueError(f"Only fixed val inference is permitted: {sample['sample_id']}")
    started = time.perf_counter()
    messages = make_messages(sample, mode, roi_entries)
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    expected_images = 1 if mode == "v2" else 2
    if video_inputs is not None or image_inputs is None or len(image_inputs) != expected_images:
        raise RuntimeError(f"Inference visual input count mismatch: {sample['sample_id']}")
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        truncation=False,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    answer_ids = generated[:, inputs["input_ids"].shape[-1] :]
    answer = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    parsed = parse_response(answer)
    grid = inputs["image_grid_thw"].detach().cpu().tolist()
    visual_tokens = [row[0] * row[1] * row[2] // 4 for row in grid]
    return {
        "stage": stage,
        "mode": mode,
        "sample_id": sample["sample_id"],
        "split": "val",
        "true_class_for_metrics_only": sample["class"],
        "raw_answer": answer,
        "parsed_prediction": parsed,
        "parse_status": "parsed" if parsed else "parse_failed",
        "correct": parsed == sample["class"],
        "prompt": PROMPTS[mode],
        "prompt_ground_truth_free": True,
        "image_input": INPUT_DESCRIPTIONS[mode],
        "input_image_count": expected_images,
        "crop_xyxy_half_open": roi_entries[sample["sample_id"]][
            "crop_xyxy_half_open"
        ],
        "vision_grid_thw": grid,
        "visual_tokens_per_image": visual_tokens,
        "total_visual_tokens": sum(visual_tokens),
        "input_token_count": int(inputs["input_ids"].shape[-1]),
        "generated_token_count": int(answer_ids.shape[-1]),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def evaluate_val(
    model,
    processor,
    samples: list[dict],
    mode: str,
    roi_entries: dict,
    stage: str,
    path: Path,
) -> dict:
    model.eval()
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records = []
    try:
        for index, sample in enumerate(samples, 1):
            records.append(
                infer_one(model, processor, sample, mode, roi_entries, stage)
            )
            if index % 50 == 0 or index == len(samples):
                print(
                    json.dumps(
                        {"stage": stage, "val_progress": f"{index}/{len(samples)}"},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                write_json(path, {"stage": stage, "completed": index, "records": records})
    finally:
        model.config.use_cache = previous_cache
    result = {
        "stage": stage,
        "records": records,
        "metrics": metrics(records),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
    write_json(path, result)
    return result


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "val_accuracy",
    "val_macro_f1",
    "malignant_recall",
    "benign_recall",
    "parse_failure_count",
    "best_epoch",
    "best_macro_f1",
    "training_seconds",
    "validation_seconds",
    "training_peak_allocated_mib",
    "validation_peak_allocated_mib",
    "checkpoint_path",
]


def append_metric(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in METRIC_FIELDS})
        handle.flush()


def balanced_smoke_subset(samples: list[dict], per_class: int) -> list[dict]:
    chosen = []
    for label in ("benign", "malignant"):
        group = sorted(
            (sample for sample in samples if sample["class"] == label),
            key=lambda sample: sample["sample_id"],
        )[:per_class]
        if len(group) != per_class:
            raise RuntimeError(f"Not enough {label} samples for smoke test")
        chosen.extend(group)
    return sorted(chosen, key=lambda sample: sample["sample_id"])


def load_processor_and_base():
    common = {
        "revision": BASE_REVISION,
        "cache_dir": MODEL_HUB_CACHE,
        "local_files_only": True,
    }
    processor = AutoProcessor.from_pretrained(MODEL_ID, **common)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"": 0},
        **common,
    )
    return processor, base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["v2", "v3"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be one directory name")
    run_dir = RUN_ROOT / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Output exists; refusing overwrite: {run_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not MODEL_CACHE.is_dir():
        raise FileNotFoundError(f"Local model cache missing: {MODEL_CACHE}")
    if not V1_ADAPTER.is_file() or sha256_file(V1_ADAPTER) != EXPECTED_V1_ADAPTER_SHA256:
        raise RuntimeError("Saved V1 adapter integrity check failed")

    run_dir.mkdir(parents=True)
    log_path = run_dir / "training.log"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    total_started = time.perf_counter()

    manifest = resolve_manifest()
    selection = rebuild_frozen_v1_selection(manifest)
    write_json(run_dir / "selection.json", selection)
    roi_manifest = load_roi_manifest(manifest, selection)
    write_json(run_dir / "roi_manifest.json", roi_manifest)
    roi_entries = roi_manifest["entries"]
    full_train_samples = selection["train_samples"]
    full_val_samples = selection["validation_samples"]
    if args.smoke:
        train_samples = balanced_smoke_subset(full_train_samples, per_class=2)
        val_samples = balanced_smoke_subset(full_val_samples, per_class=2)
        epochs = 1
        run_kind = "smoke"
    else:
        train_samples = full_train_samples
        val_samples = full_val_samples
        epochs = FULL_EPOCHS
        run_kind = "full"
    emit(
        log_path,
        "SELECTION "
        + json.dumps(
            {
                "mode": args.mode,
                "run_kind": run_kind,
                "train": dict(Counter(x["class"] for x in train_samples)),
                "val": dict(Counter(x["class"] for x in val_samples)),
                "full_selection_sha256": EXPECTED_SELECTION_SHA256,
            },
            ensure_ascii=False,
        ),
    )

    config = {
        "experiment": args.mode,
        "run_kind": run_kind,
        "seed": SEED,
        "model_id": MODEL_ID,
        "base_revision": BASE_REVISION,
        "model_cache": str(MODEL_CACHE),
        "input": INPUT_DESCRIPTIONS[args.mode],
        "image_order_fixed": args.mode == "v3",
        "prompt": PROMPTS[args.mode],
        "prompt_ground_truth_free": True,
        "fresh_lora_from_original_base": True,
        "v1_adapter_loaded": False,
        "v1_adapter_sha256_for_exclusion_audit_only": EXPECTED_V1_ADAPTER_SHA256,
        "epochs": epochs,
        "full_protocol_epochs": FULL_EPOCHS,
        "physical_microbatch_size": 1,
        "gradient_accumulation": EFFECTIVE_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "batch_policy": (
            "shuffle selected natural-ratio train representatives once per epoch "
            "with random.Random(seed + epoch); accumulate adjacent samples without "
            "class pairing, dropping, or resampling"
        ),
        "optimizer": "torch.optim.AdamW",
        "learning_rate": 1e-4,
        "adamw_weight_decay_default": 0.01,
        "gradient_clip_norm": 1.0,
        "gradient_checkpointing": True,
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        },
        "checkpoint_metric": "fixed-val Macro-F1 computed once over all validation samples",
        "checkpoint_tie_rule": "higher accuracy, then earlier epoch",
        "generation": {"max_new_tokens": 24, "do_sample": False},
        "roi_rule": roi_manifest["coordinate_rule"],
        "manifest_path": str(manifest),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "selection_file": str(run_dir / "selection.json"),
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "roi_manifest_file": str(run_dir / "roi_manifest.json"),
        "roi_manifest_sha256": sha256_file(run_dir / "roi_manifest.json"),
        "full_train_counts": selection["train_class_counts"],
        "full_val_counts": selection["validation_class_counts"],
        "actual_train_counts": dict(Counter(x["class"] for x in train_samples)),
        "actual_val_counts": dict(Counter(x["class"] for x in val_samples)),
        "test_used_for_training_validation_selection_or_inference": False,
        "packages": versions(),
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(run_dir / "config.json", config)
    result = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "epochs": [],
    }
    result_path = run_dir / "run_results.json"
    loss_path = run_dir / "training_loss.csv"
    loss_fields = [
        "epoch",
        "effective_batch",
        "optimizer_step",
        "sample_ids",
        "classes",
        "micro_losses",
        "effective_batch_mean_loss",
        "grad_norm_before_clip",
        "elapsed_seconds",
        "peak_allocated_mib",
    ]

    with loss_path.open("w", newline="") as loss_handle:
        loss_writer = csv.DictWriter(loss_handle, fieldnames=loss_fields)
        loss_writer.writeheader()
        loss_handle.flush()
        try:
            processor, base = load_processor_and_base()
            base.config.use_cache = False
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            base.enable_input_require_grads()
            model = get_peft_model(
                base,
                LoraConfig(
                    r=4,
                    lora_alpha=8,
                    lora_dropout=0.0,
                    target_modules=["q_proj", "v_proj"],
                    bias="none",
                ),
            )
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            result["trainable_parameters"] = sum(parameter.numel() for parameter in trainable)
            result["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
            if result["trainable_parameters"] != 1_474_560:
                raise RuntimeError(
                    f"Unexpected trainable parameter count: {result['trainable_parameters']}"
                )
            optimizer = torch.optim.AdamW(trainable, lr=1e-4)
            global_step = 0
            best_epoch = None
            best_macro = float("-inf")
            best_accuracy = float("-inf")
            best_checkpoint = None
            first_input_audit = None

            for epoch in range(1, epochs + 1):
                epoch_started = time.perf_counter()
                order = train_samples.copy()
                random.Random(SEED + epoch).shuffle(order)
                order_path = run_dir / f"epoch_{epoch:02d}_train_order.txt"
                order_path.write_text("\n".join(x["sample_id"] for x in order) + "\n")
                model.train()
                model.config.use_cache = False
                torch.cuda.reset_peak_memory_stats()
                batch_losses = []
                total_batches = (
                    len(order) + EFFECTIVE_BATCH_SIZE - 1
                ) // EFFECTIVE_BATCH_SIZE
                for offset in range(0, len(order), EFFECTIVE_BATCH_SIZE):
                    batch_index = offset // EFFECTIVE_BATCH_SIZE + 1
                    batch = order[offset : offset + EFFECTIVE_BATCH_SIZE]
                    global_step += 1
                    batch_started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    micro_losses = []
                    for sample in batch:
                        inputs, labels, input_audit = input_pair(
                            processor, sample, args.mode, roi_entries
                        )
                        if first_input_audit is None:
                            first_input_audit = {
                                "sample_id": sample["sample_id"],
                                **input_audit,
                            }
                        inputs = {key: value.to(model.device) for key, value in inputs.items()}
                        labels = labels.to(model.device)
                        output = model(**inputs, labels=labels)
                        loss = output.loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"Non-finite loss epoch={epoch} step={global_step}"
                            )
                        micro_losses.append(float(loss.detach().cpu()))
                        (loss / len(batch)).backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    torch.cuda.synchronize()
                    batch_loss = float(np.mean(micro_losses))
                    batch_losses.append(batch_loss)
                    loss_writer.writerow(
                        {
                            "epoch": epoch,
                            "effective_batch": batch_index,
                            "optimizer_step": global_step,
                            "sample_ids": json.dumps(
                                [x["sample_id"] for x in batch], ensure_ascii=False
                            ),
                            "classes": json.dumps(
                                [x["class"] for x in batch], ensure_ascii=False
                            ),
                            "micro_losses": json.dumps(micro_losses),
                            "effective_batch_mean_loss": batch_loss,
                            "grad_norm_before_clip": float(grad_norm.detach().cpu()),
                            "elapsed_seconds": round(time.perf_counter() - batch_started, 4),
                            "peak_allocated_mib": round(
                                torch.cuda.max_memory_allocated() / 1024**2, 2
                            ),
                        }
                    )
                    if batch_index % 100 == 0 or batch_index in {1, total_batches}:
                        loss_handle.flush()
                        emit(
                            log_path,
                            f"TRAIN_PROGRESS mode={args.mode} epoch={epoch} "
                            f"batch={batch_index}/{total_batches} "
                            f"optimizer_step={global_step} loss={batch_loss:.6f}",
                        )

                training_seconds = round(time.perf_counter() - epoch_started, 3)
                training_peak = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                train_loss = float(np.mean(batch_losses))
                checkpoint = run_dir / f"checkpoint-epoch-{epoch}"
                model.save_pretrained(checkpoint)
                validation = evaluate_val(
                    model,
                    processor,
                    val_samples,
                    args.mode,
                    roi_entries,
                    f"{args.mode}_{run_kind}_epoch_{epoch}",
                    run_dir / f"epoch_{epoch:02d}_val_results.json",
                )
                m = validation["metrics"]
                selection_improvement = m["macro_f1"] > best_macro or (
                    m["macro_f1"] == best_macro and m["accuracy"] > best_accuracy
                )
                if selection_improvement:
                    best_epoch = epoch
                    best_macro = m["macro_f1"]
                    best_accuracy = m["accuracy"]
                    best_checkpoint = checkpoint
                epoch_record = {
                    "epoch": epoch,
                    "train_sample_count": len(order),
                    "train_class_counts": dict(Counter(x["class"] for x in order)),
                    "optimizer_steps": total_batches,
                    "mean_training_loss": train_loss,
                    "minimum_batch_loss": min(batch_losses),
                    "maximum_batch_loss": max(batch_losses),
                    "training_seconds": training_seconds,
                    "training_peak_allocated_mib": training_peak,
                    "train_order_file": str(order_path),
                    "train_order_sha256": sha256_file(order_path),
                    "checkpoint_path": str(checkpoint),
                    "adapter_sha256": sha256_file(
                        checkpoint / "adapter_model.safetensors"
                    ),
                    "validation": validation,
                    "selected_as_best_after_validation": selection_improvement,
                    "best_epoch_after_validation": best_epoch,
                    "best_macro_f1_after_validation": best_macro,
                }
                result["epochs"].append(epoch_record)
                append_metric(
                    run_dir / "metrics.csv",
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_accuracy": m["accuracy"],
                        "val_macro_f1": m["macro_f1"],
                        "malignant_recall": m["malignant_sensitivity"],
                        "benign_recall": m["benign_specificity"],
                        "parse_failure_count": m["parse_failure_count"],
                        "best_epoch": best_epoch,
                        "best_macro_f1": best_macro,
                        "training_seconds": training_seconds,
                        "validation_seconds": validation["elapsed_seconds"],
                        "training_peak_allocated_mib": training_peak,
                        "validation_peak_allocated_mib": validation[
                            "peak_allocated_mib"
                        ],
                        "checkpoint_path": str(checkpoint),
                    },
                )
                emit(
                    log_path,
                    f"EPOCH_RESULT mode={args.mode} epoch={epoch} "
                    f"train_loss={train_loss:.6f} val_accuracy={m['accuracy']:.6f} "
                    f"val_macro_f1={m['macro_f1']:.6f} "
                    f"malignant_recall={m['malignant_sensitivity']:.6f} "
                    f"benign_recall={m['benign_specificity']:.6f} "
                    f"parse_failures={m['parse_failure_count']} "
                    f"best_epoch={best_epoch} best_macro_f1={best_macro:.6f}",
                )
                result["first_training_input_audit"] = first_input_audit
                write_json(result_path, result)

            if best_checkpoint is None or best_epoch is None:
                raise RuntimeError("No best checkpoint selected")
            result["checkpoint_selection"] = {
                "criterion": (
                    "highest fixed-val Macro-F1; exact tie uses higher accuracy then "
                    "earlier epoch"
                ),
                "selected_epoch": best_epoch,
                "selected_checkpoint_path": str(best_checkpoint),
                "selected_macro_f1": best_macro,
                "selected_accuracy": best_accuracy,
                "last_trained_epoch": epochs,
                "trained_all_requested_epochs": True,
            }
            write_json(run_dir / "best_checkpoint.json", result["checkpoint_selection"])

            del optimizer, model, base
            torch.cuda.empty_cache()
            processor, final_base = load_processor_and_base()
            final_model = PeftModel.from_pretrained(
                final_base,
                best_checkpoint,
                is_trainable=False,
                local_files_only=True,
            ).eval()
            sanity = infer_one(
                final_model,
                processor,
                val_samples[0],
                args.mode,
                roi_entries,
                "best_checkpoint_reload_sanity",
            )
            expected = next(
                row
                for row in result["epochs"][best_epoch - 1]["validation"]["records"]
                if row["sample_id"] == val_samples[0]["sample_id"]
            )
            result["best_checkpoint_reload"] = {
                "status": "success",
                "checkpoint_path": str(best_checkpoint),
                "adapter_sha256": sha256_file(
                    best_checkpoint / "adapter_model.safetensors"
                ),
                "sanity_inference": sanity,
                "matches_saved_best_prediction": (
                    sanity["raw_answer"] == expected["raw_answer"]
                    and sanity["parsed_prediction"] == expected["parsed_prediction"]
                ),
            }
            if not result["best_checkpoint_reload"]["matches_saved_best_prediction"]:
                raise RuntimeError("Reloaded best checkpoint prediction mismatch")
            result["run_status"] = "success"
        except Exception as exc:
            result.update(
                {
                    "run_status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            write_json(result_path, result)
            emit(log_path, result["traceback"])
            return 2

    result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result["total_elapsed_seconds"] = round(time.perf_counter() - total_started, 3)
    write_json(result_path, result)
    write_json(
        run_dir / "metrics_summary.json",
        {
            "mode": args.mode,
            "run_kind": run_kind,
            "epochs": [
                {"epoch": row["epoch"], **row["validation"]["metrics"]}
                for row in result["epochs"]
            ],
            "checkpoint_selection": result["checkpoint_selection"],
        },
    )
    emit(
        log_path,
        "RUN_COMPLETE "
        + json.dumps(
            {
                "status": "success",
                "mode": args.mode,
                "run_kind": run_kind,
                "epochs": len(result["epochs"]),
                "selected_epoch": result["checkpoint_selection"]["selected_epoch"],
                "total_seconds": result["total_elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
