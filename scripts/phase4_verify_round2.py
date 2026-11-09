"""Second-round, validation-only analysis of saved V1/V2 Verify outputs.

No model, image, split file, or test artifact is loaded. Review rankings use
only saved predictions and confidence scores; labels are consulted afterward
for offline metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ROUND1 = OUTPUTS / "phase4_verify_val496/aligned_v1_v2_scores.csv"
BASELINE_SOURCE = OUTPUTS / "phase3_roi_training_comparison/comparison.json"
LAMBDAS = (0.00, 0.05, 0.10, 0.15, 0.20)
TARGET_COVERAGES = (0.60, 0.70, 0.80)
C_THRESHOLDS = (0.60, 0.70, 0.74, 0.80)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict]:
    if not ROUND1.is_file() or not BASELINE_SOURCE.is_file():
        raise FileNotFoundError("Required first-round outputs are missing")
    rows = []
    with ROUND1.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "sample_id": raw["sample_id"],
                "true_label": raw["true_label"],
                "v1_prediction": raw["v1_prediction"],
                "v2_prediction": raw["v2_prediction"],
                "v1_confidence": float(raw["v1_confidence"]),
                "v2_confidence": float(raw["v2_confidence"]),
                "minimum_confidence": float(raw["minimum_confidence"]),
                "disagreement": raw["v1_prediction"] != raw["v2_prediction"],
                "v1_correct": raw["v1_prediction"] == raw["true_label"],
                "v2_correct": raw["v2_prediction"] == raw["true_label"],
                "group": raw["group"],
            })
    if len(rows) != 496 or len({row["sample_id"] for row in rows}) != 496:
        raise RuntimeError("Expected 496 unique aligned validation rows")
    counts = Counter(row["group"] for row in rows)
    expected = Counter({"both_correct": 393, "v1_only": 30, "v2_only": 48, "both_wrong": 25})
    if counts != expected:
        raise RuntimeError(f"Unexpected paired outcome counts: {counts}")
    return rows


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def distribution(values: list[float], prefix: str) -> dict:
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_std": statistics.pstdev(values),
        f"{prefix}_q10": quantile(values, 0.10),
        f"{prefix}_q25": quantile(values, 0.25),
        f"{prefix}_median": quantile(values, 0.50),
        f"{prefix}_q75": quantile(values, 0.75),
        f"{prefix}_q90": quantile(values, 0.90),
    }


def rank_rows(rows: list[dict], strategy: str, lambda_value: float | None = None) -> list[dict]:
    """Rank auto-handled samples without using true labels."""
    if strategy == "B-V1":
        return sorted(rows, key=lambda row: (-row["v1_confidence"], row["sample_id"]))
    if strategy == "B-V2":
        return sorted(rows, key=lambda row: (-row["v2_confidence"], row["sample_id"]))
    if strategy == "C":
        agreements = [row for row in rows if not row["disagreement"]]
        return sorted(agreements, key=lambda row: (-row["minimum_confidence"], row["sample_id"]))
    if strategy == "D":
        if lambda_value is None:
            raise ValueError("D requires lambda")
        return sorted(
            rows,
            key=lambda row: (
                -(row["v2_confidence"] - lambda_value * int(row["disagreement"])),
                row["sample_id"],
            ),
        )
    raise ValueError(strategy)


def prediction_key(strategy: str) -> str:
    return "v1_prediction" if strategy == "B-V1" else "v2_prediction"


def risk_curve(rows: list[dict], strategy: str, lambda_value: float | None = None) -> tuple[list[dict], list[dict]]:
    ranked = rank_rows(rows, strategy, lambda_value)
    pred_key = prediction_key(strategy)
    mistakes = 0
    curve = []
    for index, row in enumerate(ranked, 1):
        mistakes += int(row[pred_key] != row["true_label"])
        curve.append({
            "strategy": strategy,
            "lambda": "" if lambda_value is None else lambda_value,
            "automatic_count": index,
            "coverage": index / len(rows),
            "automatic_errors": mistakes,
            "selective_risk": mistakes / index,
            "selective_accuracy": 1.0 - mistakes / index,
        })
    return ranked, curve


def aurc_row(
    strategy: str,
    lambda_value: float | None,
    curve: list[dict],
    total: int,
    common_count: int,
) -> dict:
    native_risks = [point["selective_risk"] for point in curve]
    common_risks = native_risks[:common_count]
    if len(common_risks) != common_count:
        raise RuntimeError(f"{strategy} does not cover the common interval")
    native_raw = sum(native_risks) / total
    common_raw = sum(common_risks) / total
    return {
        "strategy": strategy,
        "lambda": "" if lambda_value is None else lambda_value,
        "final_automatic_prediction": "V1" if strategy == "B-V1" else "V2",
        "native_max_coverage": len(curve) / total,
        "native_raw_aurc": native_raw,
        "native_normalized_aurc": native_raw / (len(curve) / total),
        "common_min_coverage": 0.0,
        "common_max_coverage": common_count / total,
        "common_coverage_count": common_count,
        "common_raw_aurc": common_raw,
        "common_normalized_aurc": common_raw / (common_count / total),
        "integration_rule": "right-step empirical sum: sum(risk@k)*(1/N)",
    }


def selection_metrics(
    rows: list[dict],
    ranked: list[dict],
    strategy: str,
    target: float,
    lambda_value: float | None = None,
) -> dict:
    total = len(rows)
    automatic_count = int(round(target * total))
    if automatic_count > len(ranked):
        raise RuntimeError(f"{strategy} cannot reach target coverage {target}")
    automatic = ranked[:automatic_count]
    automatic_ids = {row["sample_id"] for row in automatic}
    reviewed = [row for row in rows if row["sample_id"] not in automatic_ids]
    pred_key = prediction_key(strategy)
    auto_errors = [row for row in automatic if row[pred_key] != row["true_label"]]
    final_errors = [row for row in rows if row[pred_key] != row["true_label"]]
    joint_errors = [row for row in rows if row["group"] == "both_wrong"]
    any_model_errors = [row for row in rows if row["group"] != "both_correct"]
    reviewed_ids = {row["sample_id"] for row in reviewed}

    def review_recall(cases: list[dict]) -> float:
        return sum(row["sample_id"] in reviewed_ids for row in cases) / len(cases)

    misses = sum(
        row["true_label"] == "malignant" and row[pred_key] == "benign"
        for row in automatic
    )
    return {
        "strategy": strategy,
        "lambda": "" if lambda_value is None else lambda_value,
        "target_coverage": target,
        "automatic_count": automatic_count,
        "actual_coverage": automatic_count / total,
        "review_count": total - automatic_count,
        "review_rate": 1.0 - automatic_count / total,
        "automatic_errors": len(auto_errors),
        "selective_risk": len(auto_errors) / automatic_count,
        "selective_accuracy": 1.0 - len(auto_errors) / automatic_count,
        "unreviewed_malignant_misses": misses,
        "joint_error_review_recall": review_recall(joint_errors),
        "joint_error_denominator": len(joint_errors),
        "final_prediction_error_review_recall": review_recall(final_errors),
        "final_prediction_error_denominator": len(final_errors),
        "any_model_error_review_recall": review_recall(any_model_errors),
        "any_model_error_denominator": len(any_model_errors),
    }


def binary_rank_auc(positive: list[float], negative: list[float]) -> float:
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return wins / (len(positive) * len(negative))


def disagreement_analysis(rows: list[dict], output_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    cases = []
    for row in rows:
        if not row["disagreement"]:
            continue
        cases.append({
            "sample_id": row["sample_id"],
            "true_label": row["true_label"],
            "v1_prediction": row["v1_prediction"],
            "v2_prediction": row["v2_prediction"],
            "outcome_group": row["group"],
            "v1_confidence": row["v1_confidence"],
            "v2_confidence": row["v2_confidence"],
            "v2_minus_v1_confidence": row["v2_confidence"] - row["v1_confidence"],
            "higher_confidence_model": "V2" if row["v2_confidence"] >= row["v1_confidence"] else "V1",
            "higher_confidence_model_correct": (
                row["v2_correct"] if row["v2_confidence"] >= row["v1_confidence"] else row["v1_correct"]
            ),
        })
    if len(cases) != 78:
        raise RuntimeError(f"Expected 78 disagreements, got {len(cases)}")
    write_csv(output_dir / "disagreement_cases.csv", cases)

    summaries = []
    for group in ("v1_only", "v2_only"):
        subset = [row for row in cases if row["outcome_group"] == group]
        summary = {
            "outcome_group": group,
            "count": len(subset),
            "rate_among_disagreements": len(subset) / len(cases),
            "higher_confidence_v1_count": sum(row["higher_confidence_model"] == "V1" for row in subset),
            "higher_confidence_v2_count": sum(row["higher_confidence_model"] == "V2" for row in subset),
        }
        for field in ("v1_confidence", "v2_confidence", "v2_minus_v1_confidence"):
            summary.update(distribution([row[field] for row in subset], field))
        summaries.append(summary)
    write_csv(output_dir / "disagreement_confidence_summary.csv", summaries)

    v2_correct_deltas = [row["v2_minus_v1_confidence"] for row in cases if row["outcome_group"] == "v2_only"]
    v1_correct_deltas = [row["v2_minus_v1_confidence"] for row in cases if row["outcome_group"] == "v1_only"]
    higher_correct = sum(bool(row["higher_confidence_model_correct"]) for row in cases)
    reliability = [
        {
            "rule": "always_V1_on_disagreement",
            "correct_count": 30,
            "sample_count": 78,
            "accuracy": 30 / 78,
            "label_free_decision_fields": "none",
            "note": "reference only",
        },
        {
            "rule": "always_V2_on_disagreement",
            "correct_count": 48,
            "sample_count": 78,
            "accuracy": 48 / 78,
            "label_free_decision_fields": "none",
            "note": "reference only",
        },
        {
            "rule": "choose_higher_raw_confidence",
            "correct_count": higher_correct,
            "sample_count": 78,
            "accuracy": higher_correct / 78,
            "label_free_decision_fields": "v1_confidence,v2_confidence",
            "note": "uncalibrated cross-model comparison",
        },
        {
            "rule": "delta_rank_auc_for_identifying_V2_correct",
            "correct_count": "",
            "sample_count": 78,
            "accuracy": binary_rank_auc(v2_correct_deltas, v1_correct_deltas),
            "label_free_decision_fields": "v2_confidence-v1_confidence",
            "note": "offline discrimination statistic; 0.5 is chance",
        },
    ]
    write_csv(output_dir / "disagreement_reliability.csv", reliability)
    return cases, summaries, reliability


def common_error_analysis(
    rows: list[dict], output_dir: Path
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    common = [row for row in rows if row["group"] == "both_wrong"]
    if len(common) != 25:
        raise RuntimeError(f"Expected 25 common errors, got {len(common)}")
    cases = []
    for row in common:
        record = {
            "sample_id": row["sample_id"],
            "true_label": row["true_label"],
            "shared_wrong_prediction": row["v2_prediction"],
            "v1_confidence": row["v1_confidence"],
            "v2_confidence": row["v2_confidence"],
            "minimum_confidence": row["minimum_confidence"],
            "disagreement": row["disagreement"],
            "d_score_all_lambdas": row["v2_confidence"],
        }
        for threshold in C_THRESHOLDS:
            record[f"c_review_at_{threshold:.2f}"] = row["minimum_confidence"] < threshold
        cases.append(record)
    write_csv(output_dir / "common_error_cases.csv", cases)

    confidence_summary = []
    for group in ("both_wrong", "both_correct"):
        subset = [row for row in rows if row["group"] == group]
        record = {"outcome_group": group, "count": len(subset)}
        for field in ("v1_confidence", "v2_confidence", "minimum_confidence"):
            record.update(distribution([row[field] for row in subset], field))
        confidence_summary.append(record)
    write_csv(output_dir / "common_error_confidence_summary.csv", confidence_summary)

    c_summary = []
    for threshold in C_THRESHOLDS:
        flagged = [row for row in cases if row[f"c_review_at_{threshold:.2f}"]]
        missed = [row for row in cases if not row[f"c_review_at_{threshold:.2f}"]]
        c_summary.append({
            "c_threshold": threshold,
            "joint_errors_reviewed": len(flagged),
            "joint_errors_missed": len(missed),
            "joint_error_review_recall": len(flagged) / len(cases),
            "missed_malignant_count": sum(row["true_label"] == "malignant" for row in missed),
            "missed_sample_ids": ";".join(row["sample_id"] for row in missed),
        })
    write_csv(output_dir / "common_error_c_detection.csv", c_summary)

    high_v2 = []
    for cutoff in (0.70, 0.75, 0.80, 0.85, 0.90):
        subset = [row for row in cases if row["v2_confidence"] >= cutoff]
        high_v2.append({
            "v2_high_confidence_cutoff": cutoff,
            "common_error_count": len(subset),
            "rate_of_all_common_errors": len(subset) / len(cases),
            "detected_by_v1_conf_below_0.60": sum(row["v1_confidence"] < 0.60 for row in subset),
            "detected_by_v1_conf_below_0.70": sum(row["v1_confidence"] < 0.70 for row in subset),
            "detected_by_v1_conf_below_0.74": sum(row["v1_confidence"] < 0.74 for row in subset),
            "detected_by_v1_conf_below_0.80": sum(row["v1_confidence"] < 0.80 for row in subset),
        })
    write_csv(output_dir / "common_error_high_v2_confidence.csv", high_v2)
    return cases, c_summary, high_v2, confidence_summary


def reference_points(rows: list[dict]) -> list[dict]:
    comparison = json.loads(BASELINE_SOURCE.read_text())
    points = []
    agreements = [row for row in rows if not row["disagreement"]]
    points.append({
        "reference": "A_disagreement",
        "coverage": len(agreements) / len(rows),
        "selective_risk": sum(row["v2_prediction"] != row["true_label"] for row in agreements) / len(agreements),
        "review_rate": 1.0 - len(agreements) / len(rows),
        "automatic_prediction": "shared V1/V2 prediction",
    })
    for model in comparison["models"]:
        points.append({
            "reference": f"{model['model']}_baseline",
            "coverage": 1.0,
            "selective_risk": 1.0 - model["metrics"]["accuracy"],
            "review_rate": 0.0,
            "automatic_prediction": model["input"],
        })
    return points


def make_plot(
    curves: dict[str, list[dict]],
    best_lambda: float,
    references: list[dict],
    common_max: float,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    styles = {
        "B-V1": ("#377eb8", "-"),
        "B-V2": ("#4daf4a", "-"),
        "C": ("#984ea3", "-"),
        f"D-lambda-{best_lambda:.2f}": ("#e41a1c", "-"),
    }
    labels = {
        "B-V1": "B-V1",
        "B-V2": "B-V2 / D lambda=0",
        "C": "C",
        f"D-lambda-{best_lambda:.2f}": f"D (lambda={best_lambda:.2f}, best val nAURC)",
    }
    for name, (color, linestyle) in styles.items():
        curve = curves[name]
        ax.plot(
            [point["coverage"] for point in curve],
            [point["selective_risk"] for point in curve],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=labels[name],
        )
    markers = {"A_disagreement": "D", "V1_baseline": "x", "V2_baseline": "+", "V3_baseline": "*"}
    for point in references:
        ax.scatter(
            point["coverage"], point["selective_risk"], s=70,
            marker=markers[point["reference"]],
            label=point["reference"].replace("_", " "),
        )
    ax.axvline(common_max, color="#666666", linestyle="--", linewidth=1,
               label=f"common AURC max={common_max:.2%}")
    ax.set_xlabel("Coverage (automatic handling rate)")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_coverage_round2.png", dpi=180)
    plt.close(fig)


def md_table(rows: list[dict], columns: list[tuple[str, str]], formats: dict[str, str]) -> str:
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "|" + "|".join("---:" for _ in columns) + "|",
    ]
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
    rows: list[dict],
    aurc: list[dict],
    target_rows: list[dict],
    disagreement_summaries: list[dict],
    reliability: list[dict],
    c_common: list[dict],
    high_v2: list[dict],
    common_confidence: list[dict],
    references: list[dict],
    best_lambda: float,
    output_dir: Path,
) -> None:
    n = len(rows)
    common_count = sum(not row["disagreement"] for row in rows)
    best_name = f"D(lambda={best_lambda:.2f})"
    aurc_display = []
    for row in aurc:
        name = row["strategy"]
        if name == "D":
            name = f"D(lambda={float(row['lambda']):.2f})"
        aurc_display.append({**row, "display": name})
    selected_target = []
    for row in target_rows:
        if row["strategy"] in ("B-V1", "B-V2", "C") or (
            row["strategy"] == "D" and abs(float(row["lambda"]) - best_lambda) < 1e-12
        ):
            display = row["strategy"] if row["strategy"] != "D" else best_name
            selected_target.append({**row, "display": display})

    aurc_table = md_table(
        aurc_display,
        [("display", "Strategy"), ("native_max_coverage", "Native max coverage"),
         ("native_raw_aurc", "Native AURC"),
         ("common_normalized_aurc", "Common normalized AURC")],
        {"native_max_coverage": "{:.2%}", "native_raw_aurc": "{:.5f}",
         "common_normalized_aurc": "{:.5f}"},
    )
    target_table = md_table(
        selected_target,
        [("display", "Strategy"), ("target_coverage", "Target"),
         ("actual_coverage", "Actual coverage"), ("selective_risk", "Risk"),
         ("review_rate", "Review rate"),
         ("unreviewed_malignant_misses", "Unreviewed malignant misses"),
         ("joint_error_review_recall", "Joint-error reviewed"),
         ("final_prediction_error_review_recall", "Final-prediction errors reviewed")],
        {"target_coverage": "{:.0%}", "actual_coverage": "{:.2%}",
         "selective_risk": "{:.2%}", "review_rate": "{:.2%}",
         "joint_error_review_recall": "{:.2%}",
         "final_prediction_error_review_recall": "{:.2%}"},
    )
    disagreement_table = md_table(
        disagreement_summaries,
        [("outcome_group", "Disagreement outcome"), ("count", "Count"),
         ("v1_confidence_median", "V1 median conf."),
         ("v2_confidence_median", "V2 median conf."),
         ("v2_minus_v1_confidence_median", "Median V2−V1 conf."),
         ("higher_confidence_v2_count", "V2 higher-conf. count")],
        {"v1_confidence_median": "{:.3f}", "v2_confidence_median": "{:.3f}",
         "v2_minus_v1_confidence_median": "{:.3f}"},
    )
    reliability_table = md_table(
        reliability,
        [("rule", "Rule/statistic"), ("correct_count", "Correct"),
         ("sample_count", "N"), ("accuracy", "Accuracy / AUC")],
        {"accuracy": "{:.3f}"},
    )
    common_table = md_table(
        c_common,
        [("c_threshold", "C threshold"), ("joint_errors_reviewed", "Reviewed"),
         ("joint_errors_missed", "Missed"),
         ("joint_error_review_recall", "Joint-error reviewed"),
         ("missed_malignant_count", "Missed malignant")],
        {"c_threshold": "{:.2f}", "joint_error_review_recall": "{:.2%}"},
    )
    high_table = md_table(
        high_v2,
        [("v2_high_confidence_cutoff", "V2 conf. cutoff"),
         ("common_error_count", "Common errors"),
         ("detected_by_v1_conf_below_0.60", "V1 conf.<0.60"),
         ("detected_by_v1_conf_below_0.70", "V1 conf.<0.70"),
         ("detected_by_v1_conf_below_0.74", "V1 conf.<0.74"),
         ("detected_by_v1_conf_below_0.80", "V1 conf.<0.80")],
        {"v2_high_confidence_cutoff": "{:.2f}"},
    )

    common_confidence_table = md_table(
        common_confidence,
        [("outcome_group", "Outcome group"), ("count", "Count"),
         ("v1_confidence_median", "V1 median conf."),
         ("v2_confidence_median", "V2 median conf."),
         ("minimum_confidence_median", "Median min-conf.")],
        {"v1_confidence_median": "{:.3f}", "v2_confidence_median": "{:.3f}",
         "minimum_confidence_median": "{:.3f}"},
    )
    reference_table = md_table(
        references,
        [("reference", "Reference"), ("coverage", "Coverage"),
         ("selective_risk", "Risk"), ("review_rate", "Review rate")],
        {"coverage": "{:.2%}", "selective_risk": "{:.2%}",
         "review_rate": "{:.2%}"},
    )

    bv2_aurc = next(row for row in aurc if row["strategy"] == "B-V2")
    best_d_aurc = next(row for row in aurc if row["strategy"] == "D" and abs(float(row["lambda"]) - best_lambda) < 1e-12)
    aurc_gain = bv2_aurc["common_normalized_aurc"] - best_d_aurc["common_normalized_aurc"]
    aurc_relative_gain = aurc_gain / bv2_aurc["common_normalized_aurc"]
    risk_comparisons = []
    for target in TARGET_COVERAGES:
        bv2 = next(row for row in target_rows if row["strategy"] == "B-V2" and row["target_coverage"] == target)
        best_d = next(row for row in target_rows if row["strategy"] == "D" and row["target_coverage"] == target and abs(float(row["lambda"]) - best_lambda) < 1e-12)
        risk_comparisons.append((target, bv2["selective_risk"] - best_d["selective_risk"]))
    gain_text = ", ".join(f"{target:.0%}: {gain:+.2%}" for target, gain in risk_comparisons)
    high_rule = next(row for row in reliability if row["rule"] == "choose_higher_raw_confidence")
    always_v2 = next(row for row in reliability if row["rule"] == "always_V2_on_disagreement")

    report = f"""# Verify second-round analysis (validation only)

Status: **AutoDL actually run successfully** using saved first-round CSV outputs only. No model or image was loaded, no training was performed, and no test artifact was read.

## AURC comparison

The empirical curve adds automatic cases in descending reliability order. AURC uses the right-step sum `sum(risk@k) × 1/496`. Because C can automatically handle at most the {common_count} agreement cases, the common normalized comparison integrates all strategies over coverage `0–{common_count}/{n}` ({common_count / n:.2%}) and divides by that interval width. Lower is better.

{aurc_table}

The best D setting on this validation set is **lambda={best_lambda:.2f}**. Its common normalized AURC improvement over B-V2 is **{aurc_gain:.5f}**. This is a retrospective validation comparison, not a frozen hyperparameter choice.

Reference points (single points, so AURC is not defined):

{reference_table}

## Same-coverage comparison

{target_table}

For best-D versus B-V2, `B-V2 risk − D risk` is: {gain_text}; positive values favor D. `Joint-error reviewed` uses the 25 V1/V2-common errors as denominator; `Final-prediction errors reviewed` uses errors of the strategy's provisional final model (73 V1 errors for B-V1, 55 V2 errors for B-V2/C/D). They are intentionally separate.

## Disagreement cases

There are 78 disagreements: 30 V1-only correct and 48 V2-only correct.

{disagreement_table}

{reliability_table}

Choosing the model with higher raw confidence reaches {high_rule['accuracy']:.2%} accuracy on disagreements, versus {always_v2['accuracy']:.2%} for always choosing V2. Because the two prompts/models are uncalibrated, raw confidence differences do not provide a reliable per-case selector unless this gain is both positive and independently calibrated.

## Common errors

All 25 common errors have disagreement=0, so D's lambda penalty never changes their score: for these cases D is exactly V2-confidence ranking. C can only identify them through low minimum confidence.

{common_confidence_table}

{common_table}

V2-high-confidence common errors and the number that V1 low confidence could expose:

{high_table}

The exact C-missed sample IDs are stored in `common_error_c_detection.csv` and all 25 per-case records in `common_error_cases.csv`.

## Does D justify V1 inference?

On this validation set, best-D lowers common normalized AURC versus B-V2 by {aurc_gain:.5f} ({aurc_relative_gain:.1%} relative), but it does **not** consistently dominate B-V2: risk is unchanged at 60% coverage, lower at 70%, and higher at 80%. D requires a V1 prediction for every sample merely to compute disagreement. On current evidence, the inconsistent target-coverage gain is not enough to justify adding a full V1 inference path solely for D. If V1 is already available for another reason, lambda={best_lambda:.2f} remains a validation-only candidate, not a frozen rule. These numbers do not establish generalization or clinical value.

## Artifacts

- `risk_coverage_round2.png`, `risk_coverage_curves.csv`, `aurc_comparison.csv`
- `target_coverage_comparison.csv`, `reference_points.csv`
- `disagreement_cases.csv`, `disagreement_confidence_summary.csv`, `disagreement_reliability.csv`
- `common_error_cases.csv`, `common_error_confidence_summary.csv`, `common_error_c_detection.csv`, `common_error_high_v2_confidence.csv`
- `run_audit.json`

## Limitations

All findings are confined to the same validation set used for checkpoint selection and exploratory threshold/lambda analysis. Confidence scores are uncalibrated across models. No test data was accessed, and no generalization improvement is claimed. A final rule and lambda must be frozen before any test evaluation.
"""
    (output_dir / "report.md").write_text(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="phase4_verify_round2_val496")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be one directory name")
    output_dir = OUTPUTS / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    total = len(rows)
    common_count = sum(not row["disagreement"] for row in rows)
    if common_count != 418:
        raise RuntimeError(f"Expected 418 agreement cases, got {common_count}")

    curves: dict[str, list[dict]] = {}
    rankings: dict[str, list[dict]] = {}
    all_curve_rows = []
    aurc = []
    for strategy in ("B-V1", "B-V2", "C"):
        ranked, curve = risk_curve(rows, strategy)
        rankings[strategy] = ranked
        curves[strategy] = curve
        all_curve_rows.extend(curve)
        aurc.append(aurc_row(strategy, None, curve, total, common_count))
    for lambda_value in LAMBDAS:
        ranked, curve = risk_curve(rows, "D", lambda_value)
        name = f"D-lambda-{lambda_value:.2f}"
        rankings[name] = ranked
        curves[name] = curve
        all_curve_rows.extend(curve)
        aurc.append(aurc_row("D", lambda_value, curve, total, common_count))
    if [point["selective_risk"] for point in curves["B-V2"]] != [
        point["selective_risk"] for point in curves["D-lambda-0.00"]
    ]:
        raise RuntimeError("D lambda=0 must exactly match B-V2")
    write_csv(output_dir / "risk_coverage_curves.csv", all_curve_rows)
    write_csv(output_dir / "aurc_comparison.csv", aurc)

    d_rows = [row for row in aurc if row["strategy"] == "D"]
    best_d = min(d_rows, key=lambda row: (row["common_normalized_aurc"], float(row["lambda"])))
    best_lambda = float(best_d["lambda"])

    target_rows = []
    for strategy in ("B-V1", "B-V2", "C"):
        for target in TARGET_COVERAGES:
            target_rows.append(selection_metrics(rows, rankings[strategy], strategy, target))
    for lambda_value in LAMBDAS:
        name = f"D-lambda-{lambda_value:.2f}"
        for target in TARGET_COVERAGES:
            target_rows.append(selection_metrics(rows, rankings[name], "D", target, lambda_value))
    bv2_by_target = {
        row["target_coverage"]: row for row in target_rows if row["strategy"] == "B-V2"
    }
    c_by_target = {
        row["target_coverage"]: row for row in target_rows if row["strategy"] == "C"
    }
    for row in target_rows:
        if row["strategy"] == "D":
            row["risk_delta_vs_B_V2"] = row["selective_risk"] - bv2_by_target[row["target_coverage"]]["selective_risk"]
            row["risk_delta_vs_C"] = row["selective_risk"] - c_by_target[row["target_coverage"]]["selective_risk"]
        else:
            row["risk_delta_vs_B_V2"] = ""
            row["risk_delta_vs_C"] = ""
    write_csv(output_dir / "target_coverage_comparison.csv", target_rows)

    _, disagreement_summaries, reliability = disagreement_analysis(rows, output_dir)
    _, c_common, high_v2, common_confidence = common_error_analysis(rows, output_dir)
    references = reference_points(rows)
    write_csv(output_dir / "reference_points.csv", references)
    make_plot(curves, best_lambda, references, common_count / total, output_dir)

    audit = {
        "status": "AutoDL actually run successfully",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_count": total,
        "input_file": str(ROUND1),
        "input_sha256": sha256_file(ROUND1),
        "baseline_source": str(BASELINE_SOURCE),
        "baseline_source_sha256": sha256_file(BASELINE_SOURCE),
        "models_or_images_loaded": False,
        "training_performed": False,
        "test_artifacts_read": False,
        "ground_truth_used_in_ranking_or_review_decision": False,
        "ground_truth_used_for_offline_metrics_only": True,
        "decision_fields": {
            "B-V1": ["v1_confidence"],
            "B-V2": ["v2_confidence"],
            "C": ["v1_prediction", "v2_prediction", "minimum_confidence"],
            "D": ["v1_prediction", "v2_prediction", "v2_confidence", "lambda"],
        },
        "lambdas": list(LAMBDAS),
        "target_coverages": list(TARGET_COVERAGES),
        "common_aurc_interval": [0.0, common_count / total],
        "common_aurc_count": common_count,
        "best_validation_lambda": best_lambda,
        "best_validation_lambda_is_not_frozen_for_test": True,
    }
    write_json(output_dir / "run_audit.json", audit)
    build_report(
        rows, aurc, target_rows, disagreement_summaries, reliability,
        c_common, high_v2, common_confidence, references, best_lambda, output_dir,
    )
    print(json.dumps({
        "status": "success",
        "output_dir": str(output_dir),
        "best_validation_lambda": best_lambda,
        "common_max_coverage": common_count / total,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
