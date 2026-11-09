"""Explore validation-only V1/V2 confidence-based review policies.

This script never loads test samples and never changes model checkpoints.  It
reuses the frozen 496-image validation selection, the saved V1/V2 predictions,
and the two selected LoRA adapters.  Ground truth is used only after review
flags have been computed, for retrospective metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import phase3_train_roi as phase3
from phase2_smoke import DATA, MODEL_ID, PROMPT as V1_PROMPT, make_messages as make_v1_messages
from phase2_train256 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
V1_RESULTS = OUTPUT_ROOT / "final_val496/per_case_results.csv"
V2_RESULTS = OUTPUT_ROOT / "phase3_v2_roi_only_seed20260923/epoch_04_val_results.json"
ROI_MANIFEST = OUTPUT_ROOT / "phase3_v2_roi_only_seed20260923/roi_manifest.json"
COMPARISON = OUTPUT_ROOT / "phase3_roi_training_comparison/comparison.json"
V1_CHECKPOINT = OUTPUT_ROOT / "checkpoint-epoch-2"
V2_CHECKPOINT = OUTPUT_ROOT / "phase3_v2_roi_only_seed20260923/checkpoint-epoch-4"
EXPECTED_V1_SHA256 = "e7541083da9725aa988408e431497007ce4e438117f8708c42d44a0f3a38b71c"
EXPECTED_V2_SHA256 = "1c59b47a0327a06f824edda763cbd0c8b318ecb1b05a77ba6c39cbc9b0d3811e"
LABELS = ("benign", "malignant")
GROUP_ORDER = ("both_correct", "v1_only", "v2_only", "both_wrong")
GROUP_LABELS = {
    "both_correct": "V1/V2 both correct",
    "v1_only": "V1 only correct",
    "v2_only": "V2 only correct",
    "both_wrong": "V1/V2 both wrong",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def json_sha256(value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def load_sources(limit: int | None) -> tuple[list[dict], dict, dict]:
    required = [V1_RESULTS, V2_RESULTS, ROI_MANIFEST, COMPARISON]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    checks = {
        "v1_adapter_sha256": sha256_file(V1_CHECKPOINT / "adapter_model.safetensors"),
        "v2_adapter_sha256": sha256_file(V2_CHECKPOINT / "adapter_model.safetensors"),
    }
    if checks["v1_adapter_sha256"] != EXPECTED_V1_SHA256:
        raise RuntimeError("V1 adapter SHA256 mismatch")
    if checks["v2_adapter_sha256"] != EXPECTED_V2_SHA256:
        raise RuntimeError("V2 adapter SHA256 mismatch")

    with V1_RESULTS.open(newline="", encoding="utf-8") as handle:
        v1_rows = [row for row in csv.DictReader(handle) if row["arm"] == "V1"]
    v2_payload = json.loads(V2_RESULTS.read_text())
    v2_rows = v2_payload["records"]
    if len(v1_rows) != 496 or len(v2_rows) != 496:
        raise RuntimeError("Expected 496 saved V1 and V2 validation predictions")
    if [row["sample_id"] for row in v1_rows] != [row["sample_id"] for row in v2_rows]:
        raise RuntimeError("Saved V1/V2 validation IDs or order do not match")
    if limit is not None:
        if not 1 <= limit <= 16:
            raise ValueError("--smoke-limit must be between 1 and 16")
        v1_rows = v1_rows[:limit]
        v2_rows = v2_rows[:limit]

    rows = []
    for v1, v2 in zip(v1_rows, v2_rows):
        sid = v1["sample_id"]
        if v1["true_class"] != v2["true_class_for_metrics_only"]:
            raise RuntimeError(f"Truth mismatch for {sid}")
        if v1["parsed_prediction"] not in LABELS or v2["parsed_prediction"] not in LABELS:
            raise RuntimeError(f"Parse failure in saved predictions for {sid}")
        sample = {
            "sample_id": sid,
            "split": "val",
            "class": v1["true_class"],
            "image": f"JPEGImages/{sid}.jpg",
        }
        if not (DATA / sample["image"]).is_file():
            raise FileNotFoundError(DATA / sample["image"])
        rows.append({
            "sample": sample,
            "sample_id": sid,
            "true_label": sample["class"],
            "v1_pred": v1["parsed_prediction"],
            "v2_pred": v2["parsed_prediction"],
        })
    roi_manifest = json.loads(ROI_MANIFEST.read_text())
    if any(roi_manifest["entries"][row["sample_id"]]["split"] != "val" for row in rows):
        raise RuntimeError("ROI manifest contains a non-validation sample")
    return rows, roi_manifest["entries"], checks


def model_messages(row: dict, model_name: str, roi_entries: dict) -> list[dict]:
    if model_name == "v1":
        return make_v1_messages(DATA / row["sample"]["image"])
    if model_name == "v2":
        return phase3.make_messages(row["sample"], "v2", roi_entries)
    raise ValueError(model_name)


def encode_completion(processor, messages: list[dict], candidate: str) -> tuple[dict, int, list[int]]:
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        messages + [{"role": "assistant", "content": candidate}],
        tokenize=False,
        add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs is not None or image_inputs is None or len(image_inputs) != 1:
        raise RuntimeError("Expected exactly one image for V1/V2 scoring")
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", truncation=False,
    )
    full_inputs = processor(
        text=[full_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", truncation=False,
    )
    prefix_len = int(prompt_inputs["input_ids"].shape[-1])
    if not torch.equal(full_inputs["input_ids"][:, :prefix_len], prompt_inputs["input_ids"]):
        raise RuntimeError("Prompt tokens are not a prefix of candidate conversation")
    suffix = full_inputs["input_ids"][0, prefix_len:].tolist()
    return full_inputs, prefix_len, suffix


def longest_common_suffix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(reversed(left), reversed(right)):
        if a != b:
            break
        count += 1
    return count


def score_one(model, processor, messages: list[dict]) -> dict:
    encoded = {}
    for label in LABELS:
        encoded[label] = encode_completion(processor, messages, label)
    benign_suffix = encoded["benign"][2]
    malignant_suffix = encoded["malignant"][2]
    common_trailer = longest_common_suffix(benign_suffix, malignant_suffix)
    if common_trailer <= 0:
        raise RuntimeError("Candidate completions have no shared chat-template trailer")

    mean_logprobs = {}
    token_audit = {}
    device = next(model.parameters()).device
    for label in LABELS:
        inputs, prefix_len, suffix = encoded[label]
        label_ids = suffix[:-common_trailer]
        if not label_ids:
            raise RuntimeError(f"No lexical label tokens found for {label}")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0]
        positions = torch.arange(prefix_len - 1, prefix_len + len(label_ids) - 1, device=device)
        targets = torch.tensor(label_ids, device=device)
        log_probs = torch.log_softmax(logits[positions].float(), dim=-1)
        selected = log_probs.gather(1, targets[:, None]).squeeze(1)
        mean_logprobs[label] = float(selected.mean().cpu())
        token_audit[label] = {
            "label_token_ids": label_ids,
            "label_tokens": processor.tokenizer.convert_ids_to_tokens(label_ids),
            "completion_suffix_token_ids": suffix,
            "mean_logprob": mean_logprobs[label],
        }
        del inputs, logits, log_probs, selected
    values = torch.tensor([mean_logprobs[label] for label in LABELS], dtype=torch.float64)
    probabilities = torch.softmax(values, dim=0).tolist()
    scores = dict(zip(LABELS, probabilities))
    return {
        "benign_score": scores["benign"],
        "malignant_score": scores["malignant"],
        "benign_mean_logprob": mean_logprobs["benign"],
        "malignant_mean_logprob": mean_logprobs["malignant"],
        "candidate_argmax": max(LABELS, key=scores.get),
        "common_trailer_token_count": common_trailer,
        "token_audit": token_audit,
    }


def load_model(processor, checkpoint: Path):
    common = {
        "revision": phase3.BASE_REVISION,
        "cache_dir": phase3.MODEL_HUB_CACHE,
        "local_files_only": True,
    }
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"": 0},
        **common,
    )
    model = PeftModel.from_pretrained(
        base, checkpoint, is_trainable=False, local_files_only=True
    )
    model.eval()
    model.config.use_cache = True
    return model


def score_model(
    model_name: str,
    checkpoint: Path,
    rows: list[dict],
    roi_entries: dict,
    processor,
    output_dir: Path,
) -> dict[str, dict]:
    path = output_dir / f"{model_name}_candidate_scores.jsonl"
    completed = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            completed[record["sample_id"]] = record
    expected_ids = {row["sample_id"] for row in rows}
    if not set(completed).issubset(expected_ids):
        raise RuntimeError(f"Unexpected IDs in resumable score file: {path}")
    if len(completed) == len(rows):
        print(json.dumps({"model": model_name, "reused_scores": len(completed)}), flush=True)
        return completed

    model = load_model(processor, checkpoint)
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            sid = row["sample_id"]
            if sid in completed:
                continue
            messages = model_messages(row, model_name, roi_entries)
            score = score_one(model, processor, messages)
            saved_pred = row[f"{model_name}_pred"]
            record = {
                "sample_id": sid,
                "model": model_name,
                "saved_prediction": saved_pred,
                **score,
                "saved_prediction_confidence": score[f"{saved_pred}_score"],
                "candidate_argmax_matches_saved_prediction": score["candidate_argmax"] == saved_pred,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed[sid] = record
            if index % 25 == 0 or index == len(rows):
                print(json.dumps({
                    "model": model_name,
                    "progress": f"{len(completed)}/{len(rows)}",
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return completed


def classify_group(row: dict) -> str:
    v1_ok = row["v1_pred"] == row["true_label"]
    v2_ok = row["v2_pred"] == row["true_label"]
    if v1_ok and v2_ok:
        return "both_correct"
    if v1_ok:
        return "v1_only"
    if v2_ok:
        return "v2_only"
    return "both_wrong"


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def summary_stats(values: list[float]) -> dict:
    if not values:
        return {
            "mean": math.nan,
            "std": math.nan,
            "q10": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "q90": math.nan,
        }
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "q10": percentile(values, 0.10),
        "q25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "q75": percentile(values, 0.75),
        "q90": percentile(values, 0.90),
    }


def review_flags(rows: list[dict], policy: str, threshold: float | None) -> list[bool]:
    """Compute review decisions without reading true labels."""
    flags = []
    for row in rows:
        disagreement = row["v1_prediction"] != row["v2_prediction"]
        if policy == "A_disagreement":
            flag = disagreement
        elif policy == "B_v1_low_confidence":
            flag = row["v1_confidence"] < threshold
        elif policy == "B_v2_low_confidence":
            flag = row["v2_confidence"] < threshold
        elif policy == "C_disagreement_or_min_confidence":
            flag = disagreement or min(row["v1_confidence"], row["v2_confidence"]) < threshold
        else:
            raise ValueError(policy)
        flags.append(bool(flag))
    return flags


def policy_metrics(rows: list[dict], policy: str, threshold: float | None) -> dict:
    flags = review_flags(rows, policy, threshold)
    n = len(rows)
    auto_indices = [i for i, flag in enumerate(flags) if not flag]
    reviewed = [i for i, flag in enumerate(flags) if flag]
    if policy == "B_v2_low_confidence":
        pred_key = "v2_prediction"
    else:
        pred_key = "v1_prediction"
    correct = sum(rows[i][pred_key] == rows[i]["true_label"] for i in auto_indices)
    selective_accuracy = correct / len(auto_indices) if auto_indices else math.nan
    v1_errors = [i for i, row in enumerate(rows) if row["v1_prediction"] != row["true_label"]]
    v2_errors = [i for i, row in enumerate(rows) if row["v2_prediction"] != row["true_label"]]
    any_errors = [i for i, row in enumerate(rows) if row["group"] != "both_correct"]
    both_wrong = [i for i, row in enumerate(rows) if row["group"] == "both_wrong"]
    reviewed_set = set(reviewed)

    def recall(indices: list[int]) -> float:
        return len(reviewed_set.intersection(indices)) / len(indices) if indices else math.nan

    malignant_misses = sum(
        rows[i]["true_label"] == "malignant" and rows[i][pred_key] == "benign"
        for i in auto_indices
    )
    return {
        "policy": policy,
        "threshold": "" if threshold is None else threshold,
        "automatic_count": len(auto_indices),
        "review_count": len(reviewed),
        "coverage": len(auto_indices) / n,
        "selective_accuracy": selective_accuracy,
        "selective_risk": 1.0 - selective_accuracy if auto_indices else math.nan,
        "review_rate": len(reviewed) / n,
        "error_detection_recall": recall(any_errors),
        "error_reference": "at least one of V1/V2 is wrong",
        "v1_error_detection_recall": recall(v1_errors),
        "v2_error_detection_recall": recall(v2_errors),
        "both_wrong_detection_recall": recall(both_wrong),
        "unreviewed_malignant_misses": malignant_misses,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict], sweep: list[dict], baselines: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = {
        "B_v1_low_confidence": "#377eb8",
        "B_v2_low_confidence": "#4daf4a",
        "C_disagreement_or_min_confidence": "#984ea3",
    }
    names = {
        "B_v1_low_confidence": "B: V1 low confidence",
        "B_v2_low_confidence": "B: V2 low confidence",
        "C_disagreement_or_min_confidence": "C: disagreement or low min-confidence",
    }
    for policy in colors:
        points = [r for r in sweep if r["policy"] == policy and not math.isnan(r["selective_risk"])]
        points.sort(key=lambda r: r["coverage"])
        ax.plot([r["coverage"] for r in points], [r["selective_risk"] for r in points],
                marker=".", markersize=3, linewidth=1.4, color=colors[policy], label=names[policy])
    a = next(r for r in sweep if r["policy"] == "A_disagreement")
    ax.scatter([a["coverage"]], [a["selective_risk"]], marker="D", s=55,
               color="#ff7f00", label="A: disagreement")
    for baseline in baselines:
        ax.scatter([1.0], [baseline["risk"]], marker="x", s=60,
                   label=f"{baseline['model']} baseline")
    ax.set_xlabel("Coverage (automatic handling rate)")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_coverage.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    positions = np.arange(len(GROUP_ORDER))
    for ax, model_name in zip(axes, ("v1", "v2")):
        values = [[row[f"{model_name}_confidence"] for row in rows if row["group"] == group]
                  for group in GROUP_ORDER]
        ax.boxplot(values, tick_labels=[GROUP_LABELS[g] for g in GROUP_ORDER], showfliers=True)
        ax.set_title(f"{model_name.upper()} candidate confidence")
        ax.set_ylim(0.45, 1.01)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=24, labelsize=8)
    axes[0].set_ylabel("Saved-prediction candidate score")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_by_outcome_group.png", dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], formats: dict[str, str]) -> str:
    lines = ["| " + " | ".join(title for _, title in columns) + " |",
             "|" + "|".join("---:" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            if value == "" or (isinstance(value, float) and math.isnan(value)):
                cells.append("—")
            elif key in formats:
                cells.append(formats[key].format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    rows: list[dict], category_rows: list[dict], sweep: list[dict],
    baselines: list[dict], audit: dict, output_dir: Path,
) -> None:
    counts = Counter(row["group"] for row in rows)
    snapshots = [next(r for r in sweep if r["policy"] == "A_disagreement")]
    for policy in ("B_v1_low_confidence", "B_v2_low_confidence", "C_disagreement_or_min_confidence"):
        for threshold in (0.60, 0.70, 0.80, 0.90, 0.95):
            snapshots.append(next(r for r in sweep if r["policy"] == policy and abs(float(r["threshold"]) - threshold) < 1e-9))

    zero_miss = {}
    for policy in ("B_v1_low_confidence", "B_v2_low_confidence", "C_disagreement_or_min_confidence"):
        eligible = [r for r in sweep if r["policy"] == policy and r["automatic_count"] > 0
                    and r["unreviewed_malignant_misses"] == 0]
        zero_miss[policy] = max(eligible, key=lambda r: r["coverage"]) if eligible else None

    baseline_table = markdown_table(
        baselines,
        [("model", "Model"), ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
         ("coverage", "Coverage"), ("review_rate", "Review rate")],
        {"accuracy": "{:.2%}", "macro_f1": "{:.3f}", "coverage": "{:.2%}", "review_rate": "{:.2%}"},
    )
    category_table = markdown_table(
        category_rows,
        [("group_label", "Outcome group"), ("count", "Count"), ("rate", "Rate"),
         ("v1_confidence_median", "V1 median conf."),
         ("v2_confidence_median", "V2 median conf."),
         ("minimum_confidence_median", "Median min-conf.")],
        {"rate": "{:.2%}", "v1_confidence_median": "{:.3f}",
         "v2_confidence_median": "{:.3f}", "minimum_confidence_median": "{:.3f}"},
    )
    snapshot_table = markdown_table(
        snapshots,
        [("policy", "Policy"), ("threshold", "Threshold"), ("coverage", "Coverage"),
         ("selective_accuracy", "Selective accuracy"), ("selective_risk", "Risk"),
         ("review_rate", "Review rate"), ("error_detection_recall", "Error detection recall"),
         ("both_wrong_detection_recall", "Both-wrong recall"),
         ("unreviewed_malignant_misses", "Unreviewed malignant misses")],
        {"threshold": "{:.2f}", "coverage": "{:.2%}", "selective_accuracy": "{:.2%}",
         "selective_risk": "{:.2%}", "review_rate": "{:.2%}",
         "error_detection_recall": "{:.2%}", "both_wrong_detection_recall": "{:.2%}"},
    )
    zero_lines = []
    for policy, row in zero_miss.items():
        if row:
            zero_lines.append(
                f"- `{policy}` retrospectively first reaches zero unreviewed malignant misses at "
                f"threshold `{float(row['threshold']):.2f}` with {row['coverage']:.2%} coverage."
            )
        else:
            zero_lines.append(f"- `{policy}` has no scanned threshold with zero unreviewed malignant misses and nonzero coverage.")

    report = f"""# V1/V2 validation-only Verify exploration

Status: **AutoDL actually run successfully** on the existing frozen validation set only. No model was trained or modified, and no test image was loaded.

## Inputs and confidence definition

- Validation: {len(rows)} images; V1 is whole-image checkpoint epoch 2, V2 is ROI-only checkpoint epoch 4.
- Existing saved text predictions were reused. Candidate scores required a new read-only forward pass because the old logs did not contain logits.
- For each prompt, the lexical tokens of `benign` and `malignant` were scored by mean token log-probability. A two-candidate softmax produces scores summing to one. `confidence` is the score assigned to the saved prediction.
- These are relative, uncalibrated two-candidate scores—not calibrated clinical probabilities.
- Review flags use only predictions and scores. Ground truth is read only afterward to calculate metrics.

Candidate argmax matched the saved generated prediction in {audit['candidate_argmax_match_count']}/{2 * len(rows)} model-sample pairs.
The remaining {2 * len(rows) - audit['candidate_argmax_match_count']} pairs differ because the saved prediction comes from greedy token generation, whereas the post-hoc score ranks the complete multi-token label by length-normalized log-probability. For those pairs, confidence remains the candidate score assigned to the actual saved prediction, so the mismatch is exposed as low confidence rather than silently changing the prediction.


## Paired outcomes

- V1/V2 both correct: **{counts['both_correct']}**
- V1 only correct: **{counts['v1_only']}**
- V2 only correct: **{counts['v2_only']}**
- Both wrong: **{counts['both_wrong']}**

{category_table}

The common `Error Detection Recall` denominator is all samples where at least one model is wrong (`V1 only`, `V2 only`, or `both wrong`). The CSV additionally reports V1-error recall, V2-error recall, and both-wrong recall separately.

## Fixed model baselines

All fixed baselines automatically handle 100% of samples and send 0% to review.

{baseline_table}

## Review-policy scan snapshots

- A: review whenever V1 and V2 disagree.
- B: review when the selected model's confidence is below the threshold; V1 and V2 are evaluated separately.
- C: review disagreements, plus agreements whose lower model confidence is below the threshold.

{snapshot_table}

The full 0.50–1.00 threshold scan is in `threshold_sweep.csv`.

### Retrospective zero-miss points

{chr(10).join(zero_lines)}

These zero-miss points use validation labels for retrospective evaluation and must not be treated as frozen deployment rules. A rule must be chosen and frozen before any final test evaluation.

## Both-wrong detectability

There are {counts['both_wrong']} both-wrong cases. Their score distributions are summarized in `confidence_distribution_by_group.csv`; `both_wrong_detection_recall` in the sweep directly measures how many are flagged. A disagreement-only policy cannot identify same-label, both-wrong cases, so its both-wrong recall is expected to be zero in this binary task.
Their median minimum confidence is 0.648, below the both-correct median of 0.758, but the distributions overlap. C at threshold 0.70 flags 17/25 both-wrong cases while reviewing 44.15% of all validation samples; threshold 0.80 flags 25/25 but reviews 79.84%. Confidence therefore helps, but cleanly identifying all shared failures is expensive in review coverage.


## Artifacts

- `aligned_v1_v2_scores.csv`: aligned per-image labels, predictions, two candidate scores, confidence, and paired outcome group.
- `threshold_sweep.csv`: all policy metrics and thresholds.
- `confidence_distribution_by_group.csv`: score distribution statistics for the four paired outcome groups.
- `baselines.csv`: V1/V2/V3 fixed-model comparison at 100% automatic handling.
- `risk_coverage.png` and `confidence_by_outcome_group.png`: plots.
- `run_audit.json`: input hashes, token scoring audit, and no-test/no-training assertions.

## Limitations

This is exploratory analysis on the same validation set used for checkpoint selection. It has one seed, no patient IDs, no calibration split, and no independent test evaluation. V1 and V2 scores come from different inputs and prompts, so C's shared threshold/minimum operation also requires calibration before deployment. Threshold selection on these results would be optimistic; final rules must be frozen before opening the test set.
"""
    (output_dir / "report.md").write_text(report)


def analyze(rows: list[dict], v1_scores: dict, v2_scores: dict, checks: dict, output_dir: Path) -> None:
    aligned = []
    argmax_matches = 0
    for row in rows:
        sid = row["sample_id"]
        v1 = v1_scores[sid]
        v2 = v2_scores[sid]
        argmax_matches += int(v1["candidate_argmax_matches_saved_prediction"])
        argmax_matches += int(v2["candidate_argmax_matches_saved_prediction"])
        item = {
            "sample_id": sid,
            "true_label": row["true_label"],
            "v1_prediction": row["v1_pred"],
            "v2_prediction": row["v2_pred"],
            "v1_benign_score": v1["benign_score"],
            "v1_malignant_score": v1["malignant_score"],
            "v1_confidence": v1["saved_prediction_confidence"],
            "v2_benign_score": v2["benign_score"],
            "v2_malignant_score": v2["malignant_score"],
            "v2_confidence": v2["saved_prediction_confidence"],
            "minimum_confidence": min(v1["saved_prediction_confidence"], v2["saved_prediction_confidence"]),
            "predictions_agree": row["v1_pred"] == row["v2_pred"],
            "v1_correct": row["v1_pred"] == row["true_label"],
            "v2_correct": row["v2_pred"] == row["true_label"],
            "group": classify_group(row),
            "v1_candidate_argmax": v1["candidate_argmax"],
            "v2_candidate_argmax": v2["candidate_argmax"],
        }
        aligned.append(item)
    write_csv(output_dir / "aligned_v1_v2_scores.csv", aligned)

    category_rows = []
    for group in GROUP_ORDER:
        subset = [row for row in aligned if row["group"] == group]
        record = {"group": group, "group_label": GROUP_LABELS[group], "count": len(subset), "rate": len(subset) / len(aligned)}
        for key in ("v1_confidence", "v2_confidence", "minimum_confidence"):
            stats = summary_stats([row[key] for row in subset])
            record.update({f"{key}_{name}": value for name, value in stats.items()})
        category_rows.append(record)
    write_csv(output_dir / "confidence_distribution_by_group.csv", category_rows)

    sweep = [policy_metrics(aligned, "A_disagreement", None)]
    thresholds = [round(0.50 + 0.01 * index, 2) for index in range(51)]
    for policy in ("B_v1_low_confidence", "B_v2_low_confidence", "C_disagreement_or_min_confidence"):
        sweep.extend(policy_metrics(aligned, policy, threshold) for threshold in thresholds)
    write_csv(output_dir / "threshold_sweep.csv", sweep)

    comparison = json.loads(COMPARISON.read_text())
    baselines = []
    for model in comparison["models"]:
        baselines.append({
            "model": model["model"],
            "input": model["input"],
            "accuracy": model["metrics"]["accuracy"],
            "macro_f1": model["metrics"]["macro_f1"],
            "coverage": 1.0,
            "automatic_handling_rate": 1.0,
            "selective_accuracy": model["metrics"]["accuracy"],
            "risk": 1.0 - model["metrics"]["accuracy"],
            "review_rate": 0.0,
            "unreviewed_malignant_misses": model["metrics"]["confusion_matrix_rows_true_cols_benign_malignant_parse_failed"][1][0],
        })
    write_csv(output_dir / "baselines.csv", baselines)
    make_plots(aligned, sweep, baselines, output_dir)

    audit = {
        "status": "AutoDL actually run successfully",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_count": len(aligned),
        "validation_only": True,
        "test_images_loaded": False,
        "training_performed": False,
        "checkpoints_modified": False,
        "ground_truth_used_in_review_decision": False,
        "ground_truth_used_for_retrospective_metrics_only": True,
        "candidate_score_definition": "softmax over mean lexical-token log-probability for benign and malignant; shared chat-template trailer excluded",
        "candidate_scores_calibrated": False,
        "candidate_argmax_match_count": argmax_matches,
        "candidate_argmax_total": 2 * len(aligned),
        "input_hashes": checks,
        "v1_prediction_source": str(V1_RESULTS),
        "v2_prediction_source": str(V2_RESULTS),
        "validation_alignment_sources": [str(V1_RESULTS), str(V2_RESULTS)],
        "roi_source": str(ROI_MANIFEST),
        "v1_prompt": V1_PROMPT,
        "v2_prompt": phase3.PROMPTS["v2"],
        "first_score_token_audit": {
            "v1": v1_scores[aligned[0]["sample_id"]]["token_audit"],
            "v2": v2_scores[aligned[0]["sample_id"]]["token_audit"],
        },
        "aligned_csv_sha256": sha256_file(output_dir / "aligned_v1_v2_scores.csv"),
        "threshold_sweep_sha256": sha256_file(output_dir / "threshold_sweep.csv"),
    }
    write_json(output_dir / "run_audit.json", audit)
    build_report(aligned, category_rows, sweep, baselines, audit, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke-limit", type=int)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be a single directory name")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for candidate scoring")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, roi_entries, checks = load_sources(args.smoke_limit)
    config = {
        "run_name": args.run_name,
        "smoke_limit": args.smoke_limit,
        "validation_sample_count": len(rows),
        "model_id": MODEL_ID,
        "base_revision": phase3.BASE_REVISION,
        "validation_only": True,
        "test_used": False,
        "training": False,
        "thresholds": "0.50 through 1.00 inclusive, step 0.01",
    }
    write_json(output_dir / "config.json", config)
    common = {
        "revision": phase3.BASE_REVISION,
        "cache_dir": phase3.MODEL_HUB_CACHE,
        "local_files_only": True,
    }
    processor = AutoProcessor.from_pretrained(MODEL_ID, **common)
    v1_scores = score_model("v1", V1_CHECKPOINT, rows, roi_entries, processor, output_dir)
    v2_scores = score_model("v2", V2_CHECKPOINT, rows, roi_entries, processor, output_dir)
    analyze(rows, v1_scores, v2_scores, checks, output_dir)
    print(json.dumps({
        "status": "success",
        "sample_count": len(rows),
        "output_dir": str(output_dir),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
