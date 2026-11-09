"""Summarize V1/V2/V3 on the same fixed 496-image validation set."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase2_smoke import ROOT
from phase2_train256 import metrics, sha256_file, write_json


V1_RESULTS = ROOT / "outputs/final_val496/per_case_results.csv"
DEFAULT_V2 = ROOT / "outputs/phase3_v2_roi_only_seed20260923"
DEFAULT_V3 = ROOT / "outputs/phase3_v3_whole_roi_seed20260923"
DEFAULT_OUTPUT = ROOT / "outputs/phase3_roi_training_comparison"


def load_v1() -> tuple[list[dict], dict]:
    with V1_RESULTS.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["arm"] == "V1"]
    records = [
        {
            "sample_id": row["sample_id"],
            "true_class_for_metrics_only": row["true_class"],
            "parsed_prediction": row["parsed_prediction"] or None,
            "raw_answer": row["raw_answer"],
        }
        for row in rows
    ]
    return records, {
        "model": "V1",
        "input": "Whole-only",
        "checkpoint": str(ROOT / "outputs/checkpoint-epoch-2"),
        "checkpoint_epoch": 2,
        "adapter_sha256": sha256_file(
            ROOT / "outputs/checkpoint-epoch-2/adapter_model.safetensors"
        ),
    }


def load_trained(run_dir: Path, expected_mode: str) -> tuple[list[dict], dict]:
    run = json.loads((run_dir / "run_results.json").read_text())
    summary = json.loads((run_dir / "metrics_summary.json").read_text())
    if run.get("run_status") != "success" or summary["run_kind"] != "full":
        raise RuntimeError(f"Run is not a successful full run: {run_dir}")
    if summary["mode"] != expected_mode:
        raise RuntimeError(f"Expected {expected_mode}, got {summary['mode']}: {run_dir}")
    selected = summary["checkpoint_selection"]
    epoch = selected["selected_epoch"]
    saved = json.loads((run_dir / f"epoch_{epoch:02d}_val_results.json").read_text())
    checkpoint = Path(selected["selected_checkpoint_path"])
    return saved["records"], {
        "model": expected_mode.upper(),
        "input": "ROI-only" if expected_mode == "v2" else "Whole + ROI",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": epoch,
        "adapter_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
        "run_dir": str(run_dir),
    }


def compare_to_v1(v1: list[dict], candidate: list[dict]) -> dict:
    base = {row["sample_id"]: row for row in v1}
    other = {row["sample_id"]: row for row in candidate}
    if list(base) != list(other):
        raise RuntimeError("Candidate validation IDs/order differ from V1")
    corrected = []
    new_errors = []
    unchanged_correct = []
    unchanged_wrong = []
    for sid in base:
        truth = base[sid]["true_class_for_metrics_only"]
        v1_correct = base[sid]["parsed_prediction"] == truth
        other_correct = other[sid]["parsed_prediction"] == truth
        if not v1_correct and other_correct:
            corrected.append(sid)
        elif v1_correct and not other_correct:
            new_errors.append(sid)
        elif v1_correct:
            unchanged_correct.append(sid)
        else:
            unchanged_wrong.append(sid)
    return {
        "corrected_count": len(corrected),
        "new_error_count": len(new_errors),
        "net_correct_change": len(corrected) - len(new_errors),
        "corrected_ids": corrected,
        "new_error_ids": new_errors,
        "unchanged_correct_count": len(unchanged_correct),
        "unchanged_wrong_count": len(unchanged_wrong),
    }


def compact_metrics(records: list[dict]) -> dict:
    scored = metrics(records)
    table = scored["confusion_matrix_rows_true_cols_prediction"]
    return {
        "sample_count": scored["sample_count"],
        "accuracy": scored["accuracy"],
        "macro_f1": scored["macro_f1"],
        "benign_recall": scored["benign_specificity"],
        "malignant_recall": scored["malignant_sensitivity"],
        "parse_failure_count": scored["parse_failure_count"],
        "confusion_matrix_rows_true_cols_benign_malignant_parse_failed": [
            [table[truth][prediction] for prediction in ("benign", "malignant", "parse_failed")]
            for truth in ("benign", "malignant")
        ],
    }


def plot_confusions(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    vmax = max(
        value
        for row in rows
        for matrix_row in row["metrics"][
            "confusion_matrix_rows_true_cols_benign_malignant_parse_failed"
        ]
        for value in matrix_row
    )
    for axis, row in zip(axes, rows):
        matrix = np.array(
            row["metrics"][
                "confusion_matrix_rows_true_cols_benign_malignant_parse_failed"
            ]
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                axis.text(
                    x,
                    y,
                    str(matrix[y, x]),
                    ha="center",
                    va="center",
                    color="white" if matrix[y, x] > vmax * 0.55 else "black",
                    fontsize=11,
                )
        axis.set_title(f"{row['model']}: {row['input']}")
        axis.set_xticks(range(3), ["Benign", "Malignant", "Parse fail"], rotation=20)
        axis.set_yticks(range(2), ["Benign", "Malignant"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.colorbar(image, ax=axes, shrink=0.82, label="Images")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(result: dict, path: Path) -> None:
    lines = [
        "# ROI 专项训练验证结果",
        "",
        "固定 validation 为与 V1 完全相同且顺序一致的 496 张图像（良性 123、恶性 373）。",
        "",
        "| 模型 | Accuracy | Macro-F1 | Corrected | New Errors | 良性 Recall | 恶性 Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["models"]:
        comparison = row.get("comparison_vs_v1")
        corrected = "—" if comparison is None else str(comparison["corrected_count"])
        new_errors = "—" if comparison is None else str(comparison["new_error_count"])
        m = row["metrics"]
        lines.append(
            f"| {row['model']}：{row['input']} | {pct(m['accuracy'])} | "
            f"{m['macro_f1']:.3f} | {corrected} | {new_errors} | "
            f"{pct(m['benign_recall'])} | {pct(m['malignant_recall'])} |"
        )
    lines.extend(["", "## 混淆矩阵", ""])
    for row in result["models"]:
        matrix = row["metrics"][
            "confusion_matrix_rows_true_cols_benign_malignant_parse_failed"
        ]
        lines.extend(
            [
                f"### {row['model']}：{row['input']}",
                "",
                "行是真实类别，列依次为 benign / malignant / parse_failed。",
                "",
                f"- benign：`{matrix[0]}`",
                f"- malignant：`{matrix[1]}`",
                "",
            ]
        )
    lines.extend(
        [
            "纠正与新增错误的完整 sample ID 见 `comparison.json`；混淆矩阵图见 "
            "`confusion_matrices.png`。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-run", type=Path, default=DEFAULT_V2)
    parser.add_argument("--v3-run", type=Path, default=DEFAULT_V3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists; refusing overwrite: {args.output}")

    v1, v1_info = load_v1()
    v2, v2_info = load_trained(args.v2_run, "v2")
    v3, v3_info = load_trained(args.v3_run, "v3")
    ids = [row["sample_id"] for row in v1]
    if len(ids) != 496 or len(ids) != len(set(ids)):
        raise RuntimeError("V1 reference is not 496 unique validation samples")
    models = []
    for records, info, comparison in [
        (v1, v1_info, None),
        (v2, v2_info, compare_to_v1(v1, v2)),
        (v3, v3_info, compare_to_v1(v1, v3)),
    ]:
        models.append(
            {
                **info,
                "metrics": compact_metrics(records),
                "comparison_vs_v1": comparison,
            }
        )
    result = {
        "validation": {
            "sample_count": 496,
            "class_counts": {"benign": 123, "malignant": 373},
            "same_ids_and_order_for_all_models": True,
            "sample_id_order_sha256": __import__("hashlib").sha256(
                "\n".join(ids).encode()
            ).hexdigest(),
        },
        "models": models,
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "comparison.json", result)
    write_report(result, args.output / "report.md")
    plot_confusions(models, args.output / "confusion_matrices.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
