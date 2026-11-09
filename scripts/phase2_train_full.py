"""First full-data Qwen3-VL V1 run with fixed split isolation and early stopping."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import json
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from phase2_lora_smoke import input_pair, versions
from phase2_smoke import DATA, MODEL_ID, OUT, PROMPT
from phase2_train256 import (
    UnionFind,
    evaluate_val,
    infer_one,
    sha256_file,
    write_json,
)


SEED = 20260923
MAX_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
EFFECTIVE_BATCH_SIZE = 2
MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "reports/tn5000_seed20260923_final/manifest.jsonl"
)
PRIOR_RUN = OUT.parent / "phase2_256_seed20260923_run2"
RUN_ROOT = OUT.parent


def emit(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a") as handle:
        handle.write(message + "\n")
        handle.flush()


def manifest_sample(record: dict, group: list[dict], split: str) -> dict:
    obj = record["objects"][0]
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
        "structurally_valid": record["structurally_valid"],
    }


def build_full_selection() -> dict:
    records = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    by_id = {record["sample_id"]: record for record in records}
    uf = UnionFind([record["sample_id"] for record in records])
    for key in ["file_sha256", "pixel_sha256"]:
        by_hash: dict[str, list[str]] = defaultdict(list)
        for record in records:
            by_hash[record[key]].append(record["sample_id"])
        for ids in by_hash.values():
            for sid in ids[1:]:
                uf.union(ids[0], sid)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[uf.find(record["sample_id"])].append(record)
    ordered_groups = sorted(
        (sorted(group, key=lambda item: item["sample_id"]) for group in groups.values()),
        key=lambda group: group[0]["sample_id"],
    )
    group_by_id = {
        record["sample_id"]: group for group in ordered_groups for record in group
    }

    label_conflicts = []
    train_samples = []
    train_groups_without_valid_rep = []
    test_representatives = []
    for group in ordered_groups:
        splits = {split for record in group for split in record["splits"]}
        labels = {record["objects"][0]["class_name"] for record in group}
        if len(labels) != 1:
            label_conflicts.append(
                {
                    "group_id": group[0]["sample_id"],
                    "sample_ids": [record["sample_id"] for record in group],
                    "labels": sorted(labels),
                }
            )
        if splits == {"train"}:
            eligible = [
                record
                for record in group
                if record["structurally_valid"]
                and len(record["objects"]) == 1
                and "train" in record["splits"]
            ]
            if eligible:
                representative = min(eligible, key=lambda item: item["sample_id"])
                train_samples.append(manifest_sample(representative, group, "train"))
            else:
                train_groups_without_valid_rep.append(
                    {
                        "group_id": group[0]["sample_id"],
                        "sample_ids": [record["sample_id"] for record in group],
                        "reason": "no structurally valid train representative",
                    }
                )
        if "test" in splits:
            candidates = [record for record in group if "test" in record["splits"]]
            representative = min(candidates, key=lambda item: item["sample_id"])
            test_representatives.append(manifest_sample(representative, group, "test"))
    if label_conflicts:
        raise RuntimeError(f"Exact duplicate label conflicts: {label_conflicts}")

    # Reuse the exact validation objects and order from the completed 256 run.
    previous = json.loads((PRIOR_RUN / "selection.json").read_text())
    val_samples = previous["validation_samples"]
    val_ids = [sample["sample_id"] for sample in val_samples]
    if len(val_ids) != 496 or len(val_ids) != len(set(val_ids)):
        raise RuntimeError("Previous fixed val is not the expected 496 unique IDs")
    val_checks = []
    for sample in val_samples:
        record = by_id[sample["sample_id"]]
        group = group_by_id[sample["sample_id"]]
        group_splits = {split for item in group for split in item["splits"]}
        checks = {
            "sample_id": sample["sample_id"],
            "manifest_val": "val" in record["splits"],
            "class_unchanged": sample["class"] == record["objects"][0]["class_name"],
            "image_unchanged": sample["image"] == record["image"],
            "file_sha256_unchanged": sample["file_sha256"] == record["file_sha256"],
            "pixel_sha256_unchanged": sample["pixel_sha256"] == record["pixel_sha256"],
            "group_does_not_intersect_test": "test" not in group_splits,
        }
        if not all(value for key, value in checks.items() if key != "sample_id"):
            raise RuntimeError(f"Fixed val mismatch: {checks}")
        val_checks.append(checks)

    train_samples.sort(key=lambda item: item["sample_id"])
    test_representatives.sort(key=lambda item: item["sample_id"])
    train_groups = {uf.find(sample["sample_id"]) for sample in train_samples}
    val_groups = {uf.find(sample["sample_id"]) for sample in val_samples}
    test_groups = {uf.find(sample["sample_id"]) for sample in test_representatives}
    intersections = {
        "train_val": sorted(train_groups & val_groups),
        "train_test": sorted(train_groups & test_groups),
        "val_test": sorted(val_groups & test_groups),
    }
    if any(intersections.values()):
        raise RuntimeError(f"Split group intersection is nonzero: {intersections}")
    if len(train_groups) != len(train_samples):
        raise RuntimeError("Train contains repeated exact groups")
    if len(test_groups) != len(test_representatives):
        raise RuntimeError("Test representative list contains repeated exact groups")

    # Read and hash only train/val inputs. Test remains metadata-only in this run.
    correspondence_failures = []
    for sample in train_samples + val_samples:
        image_path = DATA / sample["image"]
        record = by_id[sample["sample_id"]]
        checks = {
            "image_exists": image_path.is_file(),
            "image_stem_matches_id": image_path.stem == sample["sample_id"],
            "xml_filename_matches_image": record["xml_filename"] == image_path.name,
            "file_sha256_matches_manifest": image_path.is_file()
            and sha256_file(image_path) == record["file_sha256"],
        }
        if not all(checks.values()):
            correspondence_failures.append(
                {"sample_id": sample["sample_id"], "checks": checks}
            )
    if correspondence_failures:
        raise RuntimeError(f"Train/val correspondence failures: {correspondence_failures}")

    annotation_policy = {
        "002589": (
            "train XML/image size mismatch; structurally invalid singleton group, "
            "excluded without repair"
        ),
        "003813": (
            "train raw xmax boundary question; structurally invalid singleton group, "
            "excluded without clamping or repair"
        ),
        "004092": (
            "test XML/image size mismatch; retained and flagged in metadata-only test "
            "representatives; no test inference"
        ),
        "paper_class_count_mismatch": (
            "aggregate discrepancy cannot be assigned to sample IDs; preserve all XML "
            "labels exactly and do not relabel"
        ),
        "box_use": "Full-image V1 does not use bounding boxes as model input.",
    }
    return {
        "seed": SEED,
        "exact_duplicate_rule": (
            "Connected component under equality of encoded-file SHA256 OR decoded-RGB "
            "SHA256; reservation priority test > val > train; one representative per "
            "group; no oversampling."
        ),
        "train_samples": train_samples,
        "train_class_counts": dict(Counter(x["class"] for x in train_samples)),
        "train_groups_without_valid_rep": train_groups_without_valid_rep,
        "validation_samples": val_samples,
        "validation_class_counts": dict(Counter(x["class"] for x in val_samples)),
        "validation_order_sha256": hashlib.sha256(
            "\n".join(val_ids).encode()
        ).hexdigest(),
        "validation_matches_previous_order": True,
        "validation_checks": val_checks,
        "test_representatives": test_representatives,
        "test_class_counts": dict(Counter(x["class"] for x in test_representatives)),
        "test_images_read_or_inferred": False,
        "group_counts": {
            "all_records": len(records),
            "all_exact_groups": len(ordered_groups),
            "train": len(train_groups),
            "val": len(val_groups),
            "test": len(test_groups),
        },
        "group_intersections": intersections,
        "annotation_issue_policy": annotation_policy,
        "train_val_correspondence_checked": len(train_samples) + len(val_samples),
    }


def verify_and_copy_v0(run_dir: Path, selection: dict, current_versions: dict) -> dict:
    prior = json.loads((PRIOR_RUN / "run_results.json").read_text())
    prior_val = json.loads((PRIOR_RUN / "selection.json").read_text())[
        "validation_samples"
    ]
    checks = {
        "model_id_same": prior["config"]["model_id"] == MODEL_ID,
        "prompt_same": prior["config"]["prompt"] == PROMPT,
        "input_same": prior["config"]["input"] == "original full image only",
        "packages_same": prior["config"]["packages"] == current_versions,
        "val_ids_and_order_same": [x["sample_id"] for x in prior_val]
        == [x["sample_id"] for x in selection["validation_samples"]],
        "prior_v0_complete": len(prior["v0"]["records"]) == 496,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V0 cannot be reused: {checks}")
    source = PRIOR_RUN / "v0_val_results.json"
    target = run_dir / "v0_reused_val_results.json"
    shutil.copy2(source, target)
    return {
        "status": "reused_after_exact_checks",
        "source": str(source),
        "local_copy": str(target),
        "checks": checks,
        "metrics": prior["v0"]["metrics"],
        "elapsed_seconds_from_original_run": prior["v0"]["elapsed_seconds"],
        "peak_allocated_mib_from_original_run": prior["v0"]["peak_allocated_mib"],
    }


METRIC_FIELDS = [
    "epoch", "train_loss", "val_accuracy", "val_macro_f1",
    "malignant_sensitivity", "benign_specificity", "parse_failure_count",
    "best_epoch", "best_macro_f1", "early_stop_count", "training_seconds",
    "validation_seconds", "training_peak_allocated_mib",
    "validation_peak_allocated_mib", "checkpoint_path",
]


def append_metric(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in METRIC_FIELDS})
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--run-name", default="phase2_full_v1_seed20260923")
    args = parser.parse_args()
    if args.max_epochs != 10 or args.patience != 3:
        parser.error("This frozen run requires --max-epochs 10 and --patience 3")
    if Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be one directory name")
    run_dir = RUN_ROOT / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Output exists; refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "training.log"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    total_started = time.perf_counter()
    selection = build_full_selection()
    write_json(run_dir / "selection.json", selection)
    train_samples = selection["train_samples"]
    val_samples = selection["validation_samples"]
    current_versions = versions()
    v0_reuse = verify_and_copy_v0(run_dir, selection, current_versions)
    emit(
        log_path,
        "SELECTION "
        + json.dumps(
            {
                "train": selection["train_class_counts"],
                "val": selection["validation_class_counts"],
                "test": selection["test_class_counts"],
                "group_intersections": selection["group_intersections"],
                "v0": v0_reuse["status"],
            },
            ensure_ascii=False,
        ),
    )
    config = {
        "seed": SEED,
        "model_id": MODEL_ID,
        "input": "original full image only",
        "prompt": PROMPT,
        "prompt_ground_truth_free": True,
        "fresh_lora_from_original_base": True,
        "continued_from_phase2_256_adapter": False,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_improvement": "strictly greater val Macro-F1",
        "tie_rule": (
            "equal Macro-F1 uses higher accuracy then earlier epoch for checkpoint "
            "selection; tie never resets early-stop count"
        ),
        "physical_microbatch_size": 1,
        "gradient_accumulation": EFFECTIVE_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "batch_policy": (
            "shuffle all 3384 natural-ratio train representatives once per epoch; "
            "accumulate adjacent samples without class pairing, dropping, or resampling"
        ),
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "gradient_clip_norm": 1.0,
        "gradient_checkpointing": True,
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        },
        "checkpoint_metric": "fixed-val Macro-F1 computed once over all 496 samples",
        "test_used_for_training_validation_selection_or_inference": False,
        "packages": current_versions,
        "gpu": torch.cuda.get_device_name(0),
        "selection_file": str(run_dir / "selection.json"),
        "v0_reuse": v0_reuse,
    }
    write_json(run_dir / "config.json", config)
    result = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "selection_summary": {
            "train": selection["train_class_counts"],
            "val": selection["validation_class_counts"],
            "test": selection["test_class_counts"],
            "group_intersections": selection["group_intersections"],
            "annotation_issue_policy": selection["annotation_issue_policy"],
        },
        "v0": v0_reuse,
        "epochs": [],
    }
    result_path = run_dir / "run_results.json"
    loss_path = run_dir / "training_loss.csv"
    loss_fields = [
        "epoch", "effective_batch", "optimizer_step", "sample_ids", "classes",
        "micro_losses", "effective_batch_mean_loss", "grad_norm_before_clip",
        "elapsed_seconds", "peak_allocated_mib",
    ]
    with loss_path.open("w", newline="") as loss_handle:
        loss_writer = csv.DictWriter(loss_handle, fieldnames=loss_fields)
        loss_writer.writeheader()
        loss_handle.flush()
        try:
            processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            load_started = time.perf_counter()
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map={"": 0},
                local_files_only=True,
            )
            result["base_model_load_seconds"] = round(
                time.perf_counter() - load_started, 3
            )
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
            trainable = [p for p in model.parameters() if p.requires_grad]
            result["trainable_parameters"] = sum(p.numel() for p in trainable)
            result["total_parameters"] = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(trainable, lr=1e-4)
            global_step = 0
            best_epoch = None
            best_macro = float("-inf")
            best_accuracy = float("-inf")
            best_checkpoint = None
            early_stop_count = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                epoch_started = time.perf_counter()
                order = train_samples.copy()
                random.Random(SEED + epoch).shuffle(order)
                order_path = run_dir / f"epoch_{epoch:02d}_train_order.txt"
                order_path.write_text("\n".join(x["sample_id"] for x in order) + "\n")
                model.train()
                model.config.use_cache = False
                torch.cuda.reset_peak_memory_stats()
                batch_losses = []
                total_batches = (len(order) + EFFECTIVE_BATCH_SIZE - 1) // EFFECTIVE_BATCH_SIZE
                for offset in range(0, len(order), EFFECTIVE_BATCH_SIZE):
                    batch_index = offset // EFFECTIVE_BATCH_SIZE + 1
                    batch = order[offset : offset + EFFECTIVE_BATCH_SIZE]
                    global_step += 1
                    batch_started = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    micro_losses = []
                    for sample in batch:
                        inputs, labels = input_pair(processor, sample)
                        inputs = {
                            key: value.to(model.device) for key, value in inputs.items()
                        }
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
                            "elapsed_seconds": round(
                                time.perf_counter() - batch_started, 4
                            ),
                            "peak_allocated_mib": round(
                                torch.cuda.max_memory_allocated() / 1024**2, 2
                            ),
                        }
                    )
                    if batch_index % 100 == 0 or batch_index in {1, total_batches}:
                        loss_handle.flush()
                        emit(
                            log_path,
                            f"TRAIN_PROGRESS epoch={epoch} "
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
                    f"full_v1_epoch_{epoch}",
                    run_dir / f"epoch_{epoch:02d}_val_results.json",
                )
                m = validation["metrics"]
                strict_improvement = m["macro_f1"] > best_macro
                equal_macro = m["macro_f1"] == best_macro
                selection_improvement = strict_improvement or (
                    equal_macro and m["accuracy"] > best_accuracy
                )
                if strict_improvement:
                    early_stop_count = 0
                else:
                    early_stop_count += 1
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
                    "strict_macro_f1_improvement": strict_improvement,
                    "selected_by_tie_rule": selection_improvement and not strict_improvement,
                    "best_epoch_after_validation": best_epoch,
                    "best_macro_f1_after_validation": best_macro,
                    "early_stop_count_after_validation": early_stop_count,
                }
                result["epochs"].append(epoch_record)
                metric_row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_accuracy": m["accuracy"],
                    "val_macro_f1": m["macro_f1"],
                    "malignant_sensitivity": m["malignant_sensitivity"],
                    "benign_specificity": m["benign_specificity"],
                    "parse_failure_count": m["parse_failure_count"],
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_macro,
                    "early_stop_count": early_stop_count,
                    "training_seconds": training_seconds,
                    "validation_seconds": validation["elapsed_seconds"],
                    "training_peak_allocated_mib": training_peak,
                    "validation_peak_allocated_mib": validation["peak_allocated_mib"],
                    "checkpoint_path": str(checkpoint),
                }
                append_metric(run_dir / "metrics.csv", metric_row)
                summary_line = (
                    f"EPOCH_RESULT epoch={epoch} train_loss={train_loss:.6f} "
                    f"val_accuracy={m['accuracy']:.6f} "
                    f"val_macro_f1={m['macro_f1']:.6f} "
                    f"malignant_sensitivity={m['malignant_sensitivity']:.6f} "
                    f"benign_specificity={m['benign_specificity']:.6f} "
                    f"parse_failures={m['parse_failure_count']} "
                    f"best_epoch={best_epoch} best_macro_f1={best_macro:.6f} "
                    f"early_stop_count={early_stop_count}"
                )
                emit(log_path, summary_line)
                write_json(result_path, result)
                if early_stop_count >= EARLY_STOP_PATIENCE:
                    emit(
                        log_path,
                        f"EARLY_STOP epoch={epoch} patience={EARLY_STOP_PATIENCE} "
                        f"best_epoch={best_epoch} best_macro_f1={best_macro:.6f}",
                    )
                    result["stopped_early"] = True
                    result["stop_epoch"] = epoch
                    break
            else:
                result["stopped_early"] = False
                result["stop_epoch"] = MAX_EPOCHS

            if best_checkpoint is None or best_epoch is None:
                raise RuntimeError("No best checkpoint selected")
            result["checkpoint_selection"] = {
                "criterion": (
                    "highest fixed-val Macro-F1; exact tie uses higher accuracy then "
                    "earlier epoch; tie does not reset early stopping"
                ),
                "selected_epoch": best_epoch,
                "selected_checkpoint_path": str(best_checkpoint),
                "selected_macro_f1": best_macro,
                "selected_accuracy": best_accuracy,
                "last_trained_epoch": result["epochs"][-1]["epoch"],
                "final_default_is_best_not_last": True,
            }
            del optimizer, model, base
            torch.cuda.empty_cache()
            reload_started = time.perf_counter()
            final_base = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map={"": 0},
                local_files_only=True,
            )
            final_model = PeftModel.from_pretrained(
                final_base, best_checkpoint, is_trainable=False, local_files_only=True
            )
            final_model.eval()
            result["best_checkpoint_reload"] = {
                "status": "success",
                "elapsed_seconds": round(time.perf_counter() - reload_started, 3),
                "checkpoint_path": str(best_checkpoint),
                "adapter_sha256": sha256_file(
                    best_checkpoint / "adapter_model.safetensors"
                ),
            }
            sanity = infer_one(
                final_model, processor, val_samples[0], "best_checkpoint_reload_sanity"
            )
            expected = next(
                row
                for row in result["epochs"][best_epoch - 1]["validation"]["records"]
                if row["sample_id"] == val_samples[0]["sample_id"]
            )
            result["best_checkpoint_reload"]["sanity_inference"] = sanity
            result["best_checkpoint_reload"]["matches_saved_best_prediction"] = (
                sanity["raw_answer"] == expected["raw_answer"]
                and sanity["parsed_prediction"] == expected["parsed_prediction"]
            )
            if not result["best_checkpoint_reload"][
                "matches_saved_best_prediction"
            ]:
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
            "v0_reused": v0_reuse["metrics"],
            "epochs": [
                {"epoch": row["epoch"], **row["validation"]["metrics"]}
                for row in result["epochs"]
            ],
            "checkpoint_selection": result["checkpoint_selection"],
            "stopped_early": result["stopped_early"],
            "stop_epoch": result["stop_epoch"],
        },
    )
    emit(
        log_path,
        "RUN_COMPLETE "
        + json.dumps(
            {
                "status": "success",
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
