"""Three-epoch Qwen3-VL V0/V1 development run on 256 TN5000 train groups.

Exact duplicate groups are connected components under encoded-file SHA256 or
decoded-RGB SHA256 equality. Test groups have reservation priority, validation
groups come next, and training uses groups exclusive to train. No test image is
loaded for inference or training. This remains a development experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from phase2_lora_smoke import input_pair, versions
from phase2_smoke import DATA, MODEL_ID, OUT, PROMPT, make_messages, parse_response


SEED = 20260923
EPOCHS = 3
PER_CLASS = 128
MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "reports/tn5000_seed20260923_final/manifest.jsonl"
)
RUN_DIR = OUT.parent / "phase2_256_seed20260923"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {sid: sid for sid in ids}

    def find(self, sid: str) -> str:
        while self.parent[sid] != sid:
            self.parent[sid] = self.parent[self.parent[sid]]
            sid = self.parent[sid]
        return sid

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            keep, merge = sorted([left_root, right_root])
            self.parent[merge] = keep


def sample_record(record: dict, split: str, group: list[dict]) -> dict:
    obj = record["objects"][0]
    image_path = DATA / record["image"]
    checks = {
        "image_exists": image_path.is_file(),
        "image_stem_matches_id": image_path.stem == record["sample_id"],
        "xml_filename_matches_image": record["xml_filename"] == image_path.name,
        "file_sha256_matches_manifest": sha256_file(image_path)
        == record["file_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Image/ID correspondence failure for {record['sample_id']}: {checks}"
        )
    return {
        "sample_id": record["sample_id"],
        "split": split,
        "class": obj["class_name"],
        "raw_label_for_supervision_or_metrics_only": obj["raw_label"],
        "image": record["image"],
        "annotation": record["annotation"],
        "file_sha256": record["file_sha256"],
        "pixel_sha256": record["pixel_sha256"],
        "exact_group_id": min(item["sample_id"] for item in group),
        "exact_group_member_ids": sorted(item["sample_id"] for item in group),
        "correspondence_checks": checks,
    }


def build_selection() -> dict:
    records = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    valid = [
        record
        for record in records
        if record["structurally_valid"] and len(record["objects"]) == 1
    ]
    uf = UnionFind([record["sample_id"] for record in valid])
    for key in ["file_sha256", "pixel_sha256"]:
        by_hash: dict[str, list[str]] = defaultdict(list)
        for record in valid:
            by_hash[record[key]].append(record["sample_id"])
        for ids in by_hash.values():
            for sid in ids[1:]:
                uf.union(ids[0], sid)

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in valid:
        groups[uf.find(record["sample_id"])].append(record)
    ordered_groups = sorted(
        (sorted(group, key=lambda item: item["sample_id"]) for group in groups.values()),
        key=lambda group: group[0]["sample_id"],
    )

    group_rows = []
    label_conflicts = []
    for group in ordered_groups:
        splits = sorted({split for record in group for split in record["splits"]})
        labels = sorted({record["objects"][0]["class_name"] for record in group})
        row = {
            "group_id": group[0]["sample_id"],
            "sample_ids": [record["sample_id"] for record in group],
            "splits": splits,
            "labels": labels,
        }
        group_rows.append(row)
        if len(labels) != 1:
            label_conflicts.append(row)
    if label_conflicts:
        raise RuntimeError(f"Exact duplicate groups with label conflicts: {label_conflicts}")

    # Reservation order: test > val > train. Val groups that also occur in test
    # are excluded from model selection and fully listed instead of repaired.
    val_samples = []
    val_excluded = []
    train_pool: dict[str, list[tuple[dict, list[dict]]]] = defaultdict(list)
    for group in ordered_groups:
        splits = {split for record in group for split in record["splits"]}
        labels = {record["objects"][0]["class_name"] for record in group}
        class_name = next(iter(labels))
        val_members = [record for record in group if "val" in record["splits"]]
        train_members = [record for record in group if "train" in record["splits"]]
        if val_members:
            if "test" in splits:
                val_excluded.extend(
                    {
                        "sample_id": record["sample_id"],
                        "reason": "exact duplicate group intersects test",
                        "group_id": group[0]["sample_id"],
                        "group_member_ids": [item["sample_id"] for item in group],
                    }
                    for record in val_members
                )
            else:
                representative = min(val_members, key=lambda item: item["sample_id"])
                val_samples.append(sample_record(representative, "val", group))
                for record in val_members:
                    if record["sample_id"] != representative["sample_id"]:
                        val_excluded.append(
                            {
                                "sample_id": record["sample_id"],
                                "reason": "non-representative duplicate within val",
                                "representative_id": representative["sample_id"],
                                "group_id": group[0]["sample_id"],
                            }
                        )
        if train_members and splits == {"train"}:
            representative = min(train_members, key=lambda item: item["sample_id"])
            train_pool[class_name].append((representative, group))

    rng = random.Random(SEED)
    train_samples = []
    train_pool_counts = {name: len(pool) for name, pool in train_pool.items()}
    for class_name in ["benign", "malignant"]:
        pool = sorted(train_pool[class_name], key=lambda pair: pair[0]["sample_id"])
        if len(pool) < PER_CLASS:
            raise RuntimeError(
                f"Need {PER_CLASS} unique {class_name} train groups, found {len(pool)}"
            )
        chosen = rng.sample(pool, PER_CLASS)
        train_samples.extend(
            sample_record(record, "train", group) for record, group in chosen
        )

    train_samples.sort(key=lambda item: (item["class"], item["sample_id"]))
    val_samples.sort(key=lambda item: item["sample_id"])
    train_group_ids = {sample["exact_group_id"] for sample in train_samples}
    val_group_ids = {sample["exact_group_id"] for sample in val_samples}
    test_group_ids = {
        row["group_id"] for row in group_rows if "test" in row["splits"]
    }
    if train_group_ids & val_group_ids or train_group_ids & test_group_ids:
        raise RuntimeError("Train exact groups intersect reserved val/test groups")
    if len(train_group_ids) != 2 * PER_CLASS or len(val_group_ids) != len(val_samples):
        raise RuntimeError("Representative selection contains a repeated exact group")
    if any(sample["split"] == "test" for sample in train_samples + val_samples):
        raise RuntimeError("Test sample entered the selected train/val records")

    all_ids = {record["sample_id"] for record in records}
    valid_ids = {record["sample_id"] for record in valid}
    invalid_val = sorted(
        sid
        for record in records
        for sid in [record["sample_id"]]
        if sid in all_ids - valid_ids and "val" in record["splits"]
    )
    selection = {
        "seed": SEED,
        "exact_duplicate_rule": (
            "Connected component under equality of encoded file SHA256 OR decoded "
            "RGB SHA256. Reservation priority is test > val > train. Validation "
            "drops groups intersecting test and keeps the lexicographically smallest "
            "val ID per remaining group. Train uses only train-exclusive groups and "
            "one representative per group; sampling is without replacement."
        ),
        "test_inference_or_training_used": False,
        "train_pool_group_counts": train_pool_counts,
        "train_samples": train_samples,
        "train_class_counts": dict(Counter(x["class"] for x in train_samples)),
        "validation_samples": val_samples,
        "validation_class_counts": dict(Counter(x["class"] for x in val_samples)),
        "validation_excluded": sorted(val_excluded, key=lambda item: item["sample_id"]),
        "invalid_val_ids_excluded": invalid_val,
        "train_group_intersection_with_fixed_val": sorted(train_group_ids & val_group_ids),
        "train_group_intersection_with_test": sorted(train_group_ids & test_group_ids),
        "fixed_val_group_intersection_with_test": sorted(val_group_ids & test_group_ids),
        "group_inventory": {
            "valid_records": len(valid),
            "exact_groups": len(group_rows),
            "test_reserved_groups": len(test_group_ids),
        },
    }
    if selection["train_class_counts"] != {"benign": 128, "malignant": 128}:
        raise RuntimeError(f"Unexpected train balance: {selection['train_class_counts']}")
    return selection


def infer_one(model, processor, sample: dict, stage: str) -> dict:
    if sample["split"] != "val":
        raise ValueError(f"Only fixed val inference is permitted: {sample['sample_id']}")
    started = time.perf_counter()
    messages = make_messages(DATA / sample["image"])
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
        generated[:, inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    parsed = parse_response(answer)
    return {
        "stage": stage,
        "sample_id": sample["sample_id"],
        "split": "val",
        "true_class_for_metrics_only": sample["class"],
        "raw_answer": answer,
        "parsed_prediction": parsed,
        "parse_status": "parsed" if parsed else "parse_failed",
        "prompt": PROMPT,
        "prompt_ground_truth_free": True,
        "image_input": "original full image only",
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metrics(records: list[dict]) -> dict:
    table = {
        true: {pred: 0 for pred in ["benign", "malignant", "parse_failed"]}
        for true in ["benign", "malignant"]
    }
    for record in records:
        prediction = record["parsed_prediction"] or "parse_failed"
        table[record["true_class_for_metrics_only"]][prediction] += 1
    benign_tp = table["benign"]["benign"]
    malignant_tp = table["malignant"]["malignant"]
    benign_fp = table["malignant"]["benign"]
    malignant_fp = table["benign"]["malignant"]
    benign_fn = table["benign"]["malignant"] + table["benign"]["parse_failed"]
    malignant_fn = (
        table["malignant"]["benign"] + table["malignant"]["parse_failed"]
    )
    benign_f1 = safe_div(2 * benign_tp, 2 * benign_tp + benign_fp + benign_fn)
    malignant_f1 = safe_div(
        2 * malignant_tp, 2 * malignant_tp + malignant_fp + malignant_fn
    )
    total = len(records)
    failures = sum(row["parse_failed"] for row in table.values())
    return {
        "sample_count": total,
        "true_class_distribution": dict(
            Counter(record["true_class_for_metrics_only"] for record in records)
        ),
        "prediction_distribution": dict(
            Counter(record["parsed_prediction"] or "parse_failed" for record in records)
        ),
        "confusion_matrix_rows_true_cols_prediction": table,
        "accuracy": safe_div(benign_tp + malignant_tp, total),
        "macro_f1": (benign_f1 + malignant_f1) / 2,
        "benign_f1": benign_f1,
        "malignant_f1": malignant_f1,
        "malignant_sensitivity": safe_div(
            malignant_tp, sum(table["malignant"].values())
        ),
        "benign_specificity": safe_div(benign_tp, sum(table["benign"].values())),
        "parse_failure_count": failures,
        "parse_failure_rate": safe_div(failures, total),
        "metric_policy": (
            "Parse failures count as incorrect and as false negatives for the true "
            "class. Confusion matrix includes a separate parse_failed column."
        ),
    }


def evaluate_val(model, processor, samples: list[dict], stage: str, path: Path) -> dict:
    model.eval()
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records = []
    try:
        for index, sample in enumerate(samples, 1):
            records.append(infer_one(model, processor, sample, stage))
            if index % 50 == 0 or index == len(samples):
                print(
                    json.dumps(
                        {"stage": stage, "val_progress": f"{index}/{len(samples)}"},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                write_json(
                    path,
                    {"stage": stage, "completed": index, "records": records},
                )
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


def write_losses(path: Path, updates: list[dict]) -> None:
    fields = [
        "epoch", "optimizer_step", "benign_sample_id", "malignant_sample_id",
        "benign_loss", "malignant_loss", "effective_batch_mean_loss",
        "grad_norm_before_clip", "elapsed_seconds", "peak_allocated_mib",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in updates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs != EPOCHS:
        parser.error("This fixed development protocol requires exactly 3 epochs")
    if RUN_DIR.exists():
        raise FileExistsError(f"Output already exists; refusing overwrite: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    total_started = time.perf_counter()
    selection = build_selection()
    write_json(RUN_DIR / "selection.json", selection)
    train_samples = selection["train_samples"]
    val_samples = selection["validation_samples"]
    config = {
        "seed": SEED,
        "model_id": MODEL_ID,
        "input": "original full image only",
        "prompt": PROMPT,
        "prompt_ground_truth_free": True,
        "epochs": EPOCHS,
        "physical_microbatch_size": 1,
        "gradient_accumulation": 2,
        "effective_batch_size": 2,
        "effective_batch_composition": "one benign and one malignant",
        "optimizer_steps_per_epoch": PER_CLASS,
        "total_optimizer_steps": EPOCHS * PER_CLASS,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "gradient_clip_norm": 1.0,
        "dtype": "bfloat16 if supported, else float16",
        "gradient_checkpointing": True,
        "lora": {
            "fresh_initialization_from_original_base": True,
            "continued_from_fit32_adapter": False,
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        },
        "packages": versions(),
        "gpu": torch.cuda.get_device_name(0),
        "selection_file": str(RUN_DIR / "selection.json"),
        "test_used": False,
    }
    write_json(RUN_DIR / "config.json", config)
    result = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "selection_summary": {
            "train_class_counts": selection["train_class_counts"],
            "validation_class_counts": selection["validation_class_counts"],
            "validation_count": len(val_samples),
            "validation_excluded_count": len(selection["validation_excluded"]),
            "invalid_val_ids_excluded": selection["invalid_val_ids_excluded"],
        },
        "optimizer_updates": [],
        "epochs": [],
    }
    result_path = RUN_DIR / "run_results.json"
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        load_started = time.perf_counter()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": 0},
            local_files_only=True,
        )
        result["base_model_load_seconds"] = round(time.perf_counter() - load_started, 3)
        result["v0"] = evaluate_val(
            model, processor, val_samples, "v0_original_base",
            RUN_DIR / "v0_val_results.json",
        )
        write_json(result_path, result)

        # Attach a newly initialized LoRA directly to the original base model.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        model = get_peft_model(
            model,
            LoraConfig(
                r=4, lora_alpha=8, lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"], bias="none",
            ),
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        result["trainable_parameters"] = sum(p.numel() for p in trainable)
        result["total_parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"])
        benign = [sample for sample in train_samples if sample["class"] == "benign"]
        malignant = [sample for sample in train_samples if sample["class"] == "malignant"]
        global_step = 0

        for epoch in range(1, EPOCHS + 1):
            epoch_started = time.perf_counter()
            order_rng = random.Random(SEED + epoch)
            epoch_benign, epoch_malignant = benign.copy(), malignant.copy()
            order_rng.shuffle(epoch_benign)
            order_rng.shuffle(epoch_malignant)
            model.train()
            model.config.use_cache = False
            torch.cuda.reset_peak_memory_stats()
            epoch_losses = []
            for batch_index, (benign_sample, malignant_sample) in enumerate(
                zip(epoch_benign, epoch_malignant), 1
            ):
                global_step += 1
                batch_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                micro_losses = {}
                pair = [benign_sample, malignant_sample]
                order_rng.shuffle(pair)
                for sample in pair:
                    inputs, labels = input_pair(processor, sample)
                    inputs = {key: value.to(model.device) for key, value in inputs.items()}
                    labels = labels.to(model.device)
                    output = model(**inputs, labels=labels)
                    loss = output.loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss epoch={epoch} step={global_step}"
                        )
                    micro_losses[sample["class"]] = float(loss.detach().cpu())
                    (loss / 2).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                mean_loss = (micro_losses["benign"] + micro_losses["malignant"]) / 2
                epoch_losses.append(mean_loss)
                update = {
                    "epoch": epoch,
                    "optimizer_step": global_step,
                    "benign_sample_id": benign_sample["sample_id"],
                    "malignant_sample_id": malignant_sample["sample_id"],
                    "benign_loss": micro_losses["benign"],
                    "malignant_loss": micro_losses["malignant"],
                    "effective_batch_mean_loss": mean_loss,
                    "grad_norm_before_clip": float(grad_norm.detach().cpu()),
                    "elapsed_seconds": round(time.perf_counter() - batch_started, 4),
                    "peak_allocated_mib": round(
                        torch.cuda.max_memory_allocated() / 1024**2, 2
                    ),
                }
                result["optimizer_updates"].append(update)
                if batch_index % 16 == 0 or batch_index == 1:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "batch": f"{batch_index}/{PER_CLASS}",
                                "optimizer_step": global_step,
                                "mean_loss": round(mean_loss, 6),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    write_json(result_path, result)

            checkpoint = RUN_DIR / f"checkpoint-epoch-{epoch}"
            model.save_pretrained(checkpoint)
            epoch_record = {
                "epoch": epoch,
                "optimizer_steps": PER_CLASS,
                "mean_training_loss": float(np.mean(epoch_losses)),
                "minimum_batch_loss": min(epoch_losses),
                "maximum_batch_loss": max(epoch_losses),
                "training_seconds": round(time.perf_counter() - epoch_started, 3),
                "training_peak_allocated_mib": round(
                    torch.cuda.max_memory_allocated() / 1024**2, 2
                ),
                "checkpoint_path": str(checkpoint),
                "adapter_sha256": sha256_file(
                    checkpoint / "adapter_model.safetensors"
                ),
            }
            epoch_record["validation"] = evaluate_val(
                model, processor, val_samples, f"v1_epoch_{epoch}",
                RUN_DIR / f"epoch_{epoch:02d}_val_results.json",
            )
            result["epochs"].append(epoch_record)
            write_losses(RUN_DIR / "training_loss.csv", result["optimizer_updates"])
            write_json(result_path, result)

        best = max(
            result["epochs"],
            key=lambda row: (
                row["validation"]["metrics"]["macro_f1"],
                row["validation"]["metrics"]["accuracy"],
                -row["epoch"],
            ),
        )
        result["checkpoint_selection"] = {
            "criterion": "highest fixed-val macro_f1; accuracy then earlier epoch tie-break",
            "selected_epoch": best["epoch"],
            "selected_checkpoint_path": best["checkpoint_path"],
            "selected_macro_f1": best["validation"]["metrics"]["macro_f1"],
            "selected_accuracy": best["validation"]["metrics"]["accuracy"],
        }
        result["run_status"] = "success"
    except Exception as exc:
        result.update(
            {"run_status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
        )
        write_json(result_path, result)
        print(result["traceback"], flush=True)
        return 2

    result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    result["total_elapsed_seconds"] = round(time.perf_counter() - total_started, 3)
    write_json(result_path, result)
    metrics_summary = {
        "v0": result["v0"]["metrics"],
        "epochs": [
            {"epoch": row["epoch"], **row["validation"]["metrics"]}
            for row in result["epochs"]
        ],
        "checkpoint_selection": result["checkpoint_selection"],
    }
    write_json(RUN_DIR / "metrics_summary.json", metrics_summary)
    print(
        json.dumps(
            {
                "run_status": "success",
                "total_seconds": result["total_elapsed_seconds"],
                "selected_epoch": result["checkpoint_selection"]["selected_epoch"],
                "result_path": str(result_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
