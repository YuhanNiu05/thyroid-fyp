"""Create a short, local report from saved whole-image and ROI-only predictions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase1_roi_only"
DATA = ROOT / "data/official/TN5000_forReview/JPEGImages"


def score(rows: list[dict], key: str) -> dict:
    cm = {
        truth: {
            prediction: sum(
                x["true_class"] == truth and x[key] == prediction for x in rows
            )
            for prediction in ("benign", "malignant", None)
        }
        for truth in ("benign", "malignant")
    }
    correct = sum(x["true_class"] == x[key] for x in rows)
    f1 = []
    for label in ("benign", "malignant"):
        tp = cm[label][label]
        fp = sum(cm[other][label] for other in ("benign", "malignant") if other != label)
        fn = sum(cm[label][other] for other in ("benign", "malignant", None) if other != label)
        f1.append(2 * tp / (2 * tp + fp + fn))
    return {
        "correct": correct,
        "accuracy": correct / len(rows),
        "macro_f1": sum(f1) / 2,
        "malignant_sensitivity": cm["malignant"]["malignant"] / sum(cm["malignant"].values()),
        "benign_specificity": cm["benign"]["benign"] / sum(cm["benign"].values()),
        "parse_failures": sum(cm[c][None] for c in cm),
        "confusion_matrix": [
            [cm[truth][prediction] for prediction in ("benign", "malignant", None)]
            for truth in ("benign", "malignant")
        ],
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def plot_overview(rows: list[dict], counts: Counter) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    groups = [
        ("Benign\n(n=123)", [x for x in rows if x["true_class"] == "benign"]),
        ("Malignant\n(n=373)", [x for x in rows if x["true_class"] == "malignant"]),
        ("All\n(n=496)", rows),
    ]
    for offset, key, color, label in [(-0.18, "whole_pred", "#147d91", "Whole image"), (0.18, "roi_pred", "#e6a137", "ROI only")]:
        rates = [sum(x[key] == x["true_class"] for x in group) / len(group) * 100 for _, group in groups]
        bars = axes[0].bar([i + offset for i in range(3)], rates, width=0.34, color=color, label=label)
        axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    axes[0].set_xticks(range(3), [name for name, _ in groups])
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("Correct predictions (%)")
    axes[0].set_title("One image at a time")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    categories = ["Both\ncorrect", "Whole\nonly", "ROI\nonly", "Both\nwrong"]
    values = [counts[k] for k in ("both_correct", "whole_only", "roi_only", "both_wrong")]
    bars = axes[1].bar(categories, values, color=["#5aa469", "#147d91", "#e6a137", "#858c96"])
    axes[1].bar_label(bars, padding=3, fontsize=10)
    axes[1].set_ylim(0, 390)
    axes[1].set_ylabel("Images")
    axes[1].set_title("Which view is correct?")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("TN5000 validation: 496 paired images", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "overview.png", dpi=170)
    plt.close(fig)


def choose_example(rows: list[dict], pair: str, label: str) -> dict:
    candidates = sorted(
        (x for x in rows if x["pair"] == pair and x["true_class"] == label),
        key=lambda x: (x["roi_size_wh"][0] * x["roi_size_wh"][1], x["sample_id"]),
    )
    assert candidates, (pair, label)
    return candidates[len(candidates) // 2]


def plot_examples(rows: list[dict]) -> list[dict]:
    chosen = [
        choose_example(rows, "roi_only", "malignant"),
        choose_example(rows, "whole_only", "benign"),
        choose_example(rows, "both_wrong", "malignant"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), constrained_layout=True)
    for ix, x in enumerate(chosen):
        with Image.open(DATA / f"{x['sample_id']}.jpg") as source:
            full = source.convert("RGB")
        box = x["roi_box_xyxy"]
        crop = full.crop(tuple(box))
        left, right = axes[ix]
        left.imshow(full)
        left.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="#f97316", linewidth=2))
        right.imshow(crop)
        left.set_title(f"{x['sample_id']}  |  True: {x['true_class']}  |  Whole: {x['whole_pred']}", fontsize=11)
        right.set_title(f"ROI only: {x['roi_pred']}", fontsize=11)
        for ax in (left, right):
            ax.axis("off")
    fig.suptitle("Three examples of paired predictions", fontsize=15, weight="bold")
    fig.savefig(OUT / "case_examples.png", dpi=160)
    plt.close(fig)
    return [
        {
            "sample_id": x["sample_id"],
            "true_class": x["true_class"],
            "whole_prediction": x["whole_pred"],
            "roi_prediction": x["roi_pred"],
            "pair": x["pair"],
        }
        for x in chosen
    ]


def main() -> None:
    records = [json.loads(line) for line in (OUT / "predictions.jsonl").read_text().splitlines()]
    assert len(records) == 496 and len({x["sample_id"] for x in records}) == 496
    assert Counter(x["true_class"] for x in records) == Counter({"benign": 123, "malignant": 373})
    old = json.loads((ROOT / "results.json").read_text())["metrics"]["V1_full_image"]
    counts = Counter()
    rows = []
    for x in records:
        w, r, truth = x["whole"]["prediction"], x["roi"]["prediction"], x["true_class"]
        pair = "both_correct" if w == truth and r == truth else "whole_only" if w == truth else "roi_only" if r == truth else "both_wrong"
        counts[pair] += 1
        rows.append({**x, "whole_pred": w, "roi_pred": r, "oracle_pred": w if w == truth else r if r == truth else w, "pair": pair})
    metrics = {key: score(rows, key) for key in ("whole_pred", "roi_pred", "oracle_pred")}
    assert metrics["whole_pred"]["confusion_matrix"] == old["confusion_matrix"]
    assert metrics["whole_pred"]["correct"] == 423
    assert counts == Counter({"both_correct": 345, "whole_only": 78, "roi_only": 33, "both_wrong": 40})
    by_class = {
        label: dict(Counter(x["pair"] for x in rows if x["true_class"] == label))
        for label in ("benign", "malignant")
    }
    plot_overview(rows, counts)
    examples = plot_examples(rows)
    summary = {
        "status": "AutoDL 实际运行成功",
        "sample_count": len(rows),
        "model": "Qwen/Qwen3-VL-4B-Instruct + checkpoint-epoch-2",
        "views": "independent single-image full versus manually annotated ROI",
        "metrics": metrics,
        "paired_counts": dict(counts),
        "paired_counts_by_class": by_class,
        "oracle_accuracy_gap_vs_whole": metrics["oracle_pred"]["accuracy"] - metrics["whole_pred"]["accuracy"],
        "oracle_accuracy_gap_vs_best_fixed": metrics["oracle_pred"]["accuracy"] - max(metrics[k]["accuracy"] for k in ("whole_pred", "roi_pred")),
        "roi_malignant_predictions": sum(x["roi_pred"] == "malignant" for x in rows),
        "always_malignant_selection_oracle_rescues": sum(x["true_class"] == "malignant" and x["whole_pred"] != "malignant" for x in rows),
        "examples": examples,
        "limitations": ["Fixed validation set selected the checkpoint", "The model was fine-tuned on full images, not ROI-only images", "No patient IDs; patient-level independence cannot be verified", "Oracle uses true labels for offline analysis only", "Final test set was not used"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    w, r, o = (metrics[k] for k in ("whole_pred", "roi_pred", "oracle_pred"))
    report = f"""# 第一阶段：只看整图与只看结节 ROI

**已实际运行：**现有 Qwen3-VL 模型在同一批 496 张验证图上，每次独立看一张图。ROI 来自人工标注框。

| 输入 | 正确数 | 准确率 | Macro-F1 | 恶性敏感度 | 良性特异度 |
|---|---:|---:|---:|---:|---:|
| 仅整图 | {w['correct']}/496 | {pct(w['accuracy'])} | {w['macro_f1']:.3f} | {pct(w['malignant_sensitivity'])} | {pct(w['benign_specificity'])} |
| 仅 ROI | {r['correct']}/496 | {pct(r['accuracy'])} | {r['macro_f1']:.3f} | {pct(r['malignant_sensitivity'])} | {pct(r['benign_specificity'])} |
| 理想选择¹ | {o['correct']}/496 | {pct(o['accuracy'])} | {o['macro_f1']:.3f} | {pct(o['malignant_sensitivity'])} | {pct(o['benign_specificity'])} |

![整图和 ROI 的表现及逐例重叠](overview.png)

仅 ROI **纠正了整图的 33 个错误**，也**改错了整图原本正确的 78 例**。纠正的 33 例中，恶性 {by_class['malignant'].get('roi_only', 0)} 例、良性 {by_class['benign'].get('roi_only', 0)} 例；改错的 78 例中，良性 {by_class['benign'].get('whole_only', 0)} 例、恶性 {by_class['malignant'].get('whole_only', 0)} 例。因此，当前模型直接改用 ROI 会降低总体表现；但两个视图确实有互补答案。

理想选择最多比整图提高 **{(o['accuracy']-w['accuracy'])*100:.2f} 个百分点**。这是固定的两个已有答案之间的上限，不能当作实际 Agent 的成绩。ROI 在 496 张中有 447 张预测为恶性；作为核对，若第二个答案始终写“恶性”，离线理想选择同样能救回 34 例，甚至略多于 ROI 的 33 例。因此，**这个 oracle gap 还不能证明 ROI 提供了独特的影像信息**。

![三组逐例预测示例：橙框为人工 ROI](case_examples.png)

这三个例子按对应类别的 ROI 面积中位数选取：`{examples[0]['sample_id']}` 为 ROI 纠正恶性误判，`{examples[1]['sample_id']}` 为 ROI 改错良性判断，`{examples[2]['sample_id']}` 为两者都错。图像只展示预测差异，不解释影像征象的病理原因。

¹ 理想选择在**离线分析**时使用真实标签，从两次判断中选正确的一次；真实标签从未进入模型输入。

**结论边界：**checkpoint 由同一验证集选出，模型只接受过整图微调，缺少患者 ID，最终 test 尚未运行。这些结果说明当前模型存在可纠错样本，不保证可以学会可靠选择，也不代表经过 ROI 专项训练后的上限。
"""
    (OUT / "report.md").write_text(report)
    print(json.dumps({"metrics": metrics, "counts": dict(counts), "by_class": by_class, "examples": examples}, ensure_ascii=False, indent=2))
    print("Saved", OUT / "report.md")


if __name__ == "__main__":
    main()
