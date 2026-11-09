"""Generate a concise Chinese Word report for the two Verify analyses.

The document is written as OOXML with only the Python standard library. It
reads saved validation-only CSV/PNG artifacts and does not load models, images,
or any test-set artifact.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from xml.sax.saxutils import escape
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ROUND1 = ROOT / "outputs/phase4_verify_val496"
ROUND2 = ROOT / "outputs/phase4_verify_round2_val496"
OUTPUT = ROUND2 / "Verify实验报告_中文精简版.docx"
RISK_FIGURE = ROUND2 / "risk_coverage_round2.png"
CONFIDENCE_FIGURE = ROUND1 / "confidence_by_outcome_group.png"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: str | float, digits: int = 2) -> str:
    return f"{float(value):.{digits}%}"


def run(text: str, *, bold: bool = False, size: int = 19, color: str = "263238") -> str:
    properties = [
        '<w:rFonts w:ascii="Aptos" w:hAnsi="Aptos" w:eastAsia="微软雅黑"/>',
        f'<w:sz w:val="{size}"/>',
        f'<w:color w:val="{color}"/>',
    ]
    if bold:
        properties.append("<w:b/>")
    return (
        "<w:r><w:rPr>" + "".join(properties) + "</w:rPr>"
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    )


def paragraph(
    text: str = "",
    *,
    bold: bool = False,
    size: int = 19,
    color: str = "263238",
    align: str | None = None,
    before: int = 0,
    after: int = 90,
    keep_next: bool = False,
    indent: int = 0,
    style: str | None = None,
) -> str:
    properties = [
        f'<w:spacing w:before="{before}" w:after="{after}" w:line="280" w:lineRule="auto"/>'
    ]
    if align:
        properties.append(f'<w:jc w:val="{align}"/>')
    if keep_next:
        properties.append("<w:keepNext/>")
    if indent:
        properties.append(f'<w:ind w:left="{indent}"/>')
    if style:
        properties.append(f'<w:pStyle w:val="{style}"/>')
    return f"<w:p><w:pPr>{''.join(properties)}</w:pPr>{run(text, bold=bold, size=size, color=color)}</w:p>"


def rich_paragraph(parts: list[tuple[str, bool]], *, after: int = 90, size: int = 19) -> str:
    content = "".join(run(text, bold=bold, size=size) for text, bold in parts)
    return (
        "<w:p><w:pPr>"
        f'<w:spacing w:before="0" w:after="{after}" w:line="280" w:lineRule="auto"/>'
        f"</w:pPr>{content}</w:p>"
    )


def heading(text: str, level: int = 1) -> str:
    if level == 1:
        return paragraph(
            text, bold=True, size=26, color="145B6B", before=150, after=90,
            keep_next=True, style="Heading1",
        )
    return paragraph(
        text, bold=True, size=22, color="287687", before=110, after=70,
        keep_next=True, style="Heading2",
    )


def bullet(text: str) -> str:
    return paragraph("• " + text, size=18, after=45, indent=260)


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    if len(headers) != len(widths) or any(len(row) != len(headers) for row in rows):
        raise ValueError("Table shape mismatch")
    total = sum(widths)
    output = [
        "<w:tbl><w:tblPr>",
        f'<w:tblW w:w="{total}" w:type="dxa"/>',
        '<w:tblLayout w:type="fixed"/>',
        '<w:tblBorders><w:top w:val="single" w:sz="6" w:color="AFC3CB"/>'
        '<w:left w:val="single" w:sz="3" w:color="D6E0E4"/>'
        '<w:bottom w:val="single" w:sz="6" w:color="AFC3CB"/>'
        '<w:right w:val="single" w:sz="3" w:color="D6E0E4"/>'
        '<w:insideH w:val="single" w:sz="3" w:color="D6E0E4"/>'
        '<w:insideV w:val="single" w:sz="3" w:color="D6E0E4"/></w:tblBorders>',
        '<w:cellMar><w:top w:w="60" w:type="dxa"/><w:start w:w="70" w:type="dxa"/>'
        '<w:bottom w:w="60" w:type="dxa"/><w:end w:w="70" w:type="dxa"/></w:cellMar>',
        "</w:tblPr><w:tblGrid>",
        "".join(f'<w:gridCol w:w="{width}"/>' for width in widths),
        "</w:tblGrid>",
    ]

    def row_xml(values: list[str], header: bool, shade: bool = False) -> str:
        cells = []
        for value, width in zip(values, widths):
            fill = "DCEAF0" if header else "F4F8FA" if shade else "FFFFFF"
            cells.append(
                "<w:tc><w:tcPr>"
                f'<w:tcW w:w="{width}" w:type="dxa"/><w:shd w:fill="{fill}"/>'
                '<w:vAlign w:val="center"/></w:tcPr>'
                '<w:p><w:pPr><w:spacing w:before="10" w:after="10"/>'
                '<w:jc w:val="center"/></w:pPr>'
                + run(str(value), bold=header, size=16 if header else 15)
                + "</w:p></w:tc>"
            )
        return '<w:tr><w:trPr><w:cantSplit/></w:trPr>' + "".join(cells) + "</w:tr>"

    output.append(row_xml(headers, True))
    for index, values in enumerate(rows):
        output.append(row_xml(values, False, shade=index % 2 == 1))
    output.append("</w:tbl>")
    return "".join(output)


def callout(text: str) -> str:
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="9900" w:type="dxa"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="8" w:color="57A0AE"/>'
        '<w:left w:val="single" w:sz="8" w:color="57A0AE"/>'
        '<w:bottom w:val="single" w:sz="8" w:color="57A0AE"/>'
        '<w:right w:val="single" w:sz="8" w:color="57A0AE"/></w:tblBorders>'
        '</w:tblPr><w:tr><w:tc><w:tcPr><w:shd w:fill="EDF6F8"/>'
        '<w:tcMar><w:top w:w="120" w:type="dxa"/><w:start w:w="140" w:type="dxa"/>'
        '<w:bottom w:w="120" w:type="dxa"/><w:end w:w="140" w:type="dxa"/></w:tcMar>'
        '</w:tcPr>'
        + paragraph(text, bold=True, size=19, color="145B6B", after=0)
        + "</w:tc></w:tr></w:tbl>"
    )


def image_paragraph(rel_id: str, name: str, path: Path, width_inches: float, doc_id: int) -> str:
    with Image.open(path) as image:
        ratio = image.height / image.width
    cx = int(width_inches * 914400)
    cy = int(width_inches * ratio * 914400)
    drawing = f"""
<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
<wp:docPr id="{doc_id}" name="{escape(name)}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{escape(name)}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>
"""
    return (
        '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="70" w:after="45"/></w:pPr>'
        + drawing
        + "</w:p>"
    )


def caption(text: str) -> str:
    return paragraph(text, size=16, color="53636B", align="center", after=90, keep_next=True)


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def build_document() -> str:
    baselines = read_csv(ROUND1 / "baselines.csv")
    aligned = read_csv(ROUND1 / "aligned_v1_v2_scores.csv")
    round1_sweep = read_csv(ROUND1 / "threshold_sweep.csv")
    aurc = read_csv(ROUND2 / "aurc_comparison.csv")
    targets = read_csv(ROUND2 / "target_coverage_comparison.csv")
    disagreement = read_csv(ROUND2 / "disagreement_reliability.csv")
    common_detection = read_csv(ROUND2 / "common_error_c_detection.csv")

    group_counts = {
        group: sum(row["group"] == group for row in aligned)
        for group in ("both_correct", "v1_only", "v2_only", "both_wrong")
    }

    def sweep(policy: str, threshold: str = "") -> dict:
        return next(
            row for row in round1_sweep
            if row["policy"] == policy and row["threshold"] == threshold
        )

    selected_round1 = [
        ("A：不一致复核", sweep("A_disagreement")),
        ("B-V2：阈值 0.80", sweep("B_v2_low_confidence", "0.8")),
        ("C：阈值 0.70", sweep("C_disagreement_or_min_confidence", "0.7")),
        ("C：阈值 0.74", sweep("C_disagreement_or_min_confidence", "0.74")),
        ("C：阈值 0.80", sweep("C_disagreement_or_min_confidence", "0.8")),
    ]

    aurc_keep = []
    for row in aurc:
        if row["strategy"] in ("B-V1", "B-V2", "C"):
            aurc_keep.append(row)
        elif row["strategy"] == "D" and row["lambda"] in ("0.0", "0.05", "0.1"):
            aurc_keep.append(row)

    target_keep = [
        row for row in targets
        if row["strategy"] in ("B-V2", "C")
        or (row["strategy"] == "D" and row["lambda"] == "0.05")
    ]
    target_keep.sort(key=lambda row: (float(row["target_coverage"]), {"B-V2": 0, "C": 1, "D": 2}[row["strategy"]]))

    body = []
    body.append(paragraph("Verify 实验报告", bold=True, size=36, color="145B6B", align="center", after=40, style="Title"))
    body.append(paragraph("V1 整图 + V2 ROI：validation 选择性复核探索", bold=True, size=22, align="center", after=110))
    body.append(paragraph("报告日期：2026 年 11 月 1 日", size=17, color="53636B", align="center", after=150))
    body.append(callout(
        "结论：V2 是更强的单模型基线。联合策略 D（λ=0.05）在共同区间内的归一化 AURC 优于 B-V2，"
        "但在 60% / 70% / 80% Coverage 上表现分别为持平 / 更好 / 更差，未形成稳定支配；"
        "当前证据不足以支持仅为 D 增加完整 V1 推理成本。"
    ))
    body.append(paragraph("范围与口径", bold=True, size=21, color="287687", before=120, after=60))
    body.append(bullet("数据：固定 validation 496 张（良性 123、恶性 373）；不访问 test。"))
    body.append(bullet("模型：V1 整图、V2 人工 ROI、V3 整图 + ROI；Verify 不重新训练模型。"))
    body.append(bullet("置信度：benign / malignant 两候选的长度归一化相对分数，未经跨模型校准。"))
    body.append(bullet("Needs Review 决策只使用预测与置信度；真实标签仅用于离线评价。"))

    body.append(heading("1. 基线与样本关系"))
    body.append(table(
        ["模型", "输入", "Accuracy", "Macro-F1", "自动处理率"],
        [[row["model"], row["input"], pct(row["accuracy"]), f"{float(row['macro_f1']):.3f}", "100%"] for row in baselines],
        [1500, 2200, 1700, 1700, 1900],
    ))
    body.append(paragraph("V2 相比 V1：Accuracy +3.63 个百分点，Macro-F1 +0.053；V3 最高，但 Verify 本阶段仅使用 V1/V2。", size=17, color="53636B", after=90))
    body.append(table(
        ["V1/V2 逐例关系", "数量", "占比"],
        [
            ["都正确", str(group_counts["both_correct"]), pct(group_counts["both_correct"] / 496)],
            ["仅 V1 正确", str(group_counts["v1_only"]), pct(group_counts["v1_only"] / 496)],
            ["仅 V2 正确", str(group_counts["v2_only"]), pct(group_counts["v2_only"] / 496)],
            ["都错误", str(group_counts["both_wrong"]), pct(group_counts["both_wrong"] / 496)],
        ],
        [4200, 2200, 2200],
    ))
    body.append(image_paragraph("rIdConfidence", "confidence_by_outcome_group", CONFIDENCE_FIGURE, 6.6, 1))
    body.append(caption("图 1　四类逐例结果的 V1/V2 相对置信度分布。共同错误置信度较低，但与共同正确样本明显重叠。"))

    body.append(heading("2. 第一轮 Verify：阈值策略"))
    body.append(table(
        ["策略", "Coverage", "选择性准确率", "复核率", "错误检出率", "共同错误复核", "未复核恶性漏诊"],
        [[
            name,
            pct(row["coverage"]),
            pct(row["selective_accuracy"]),
            pct(row["review_rate"]),
            pct(row["error_detection_recall"]),
            pct(row["both_wrong_detection_recall"]),
            row["unreviewed_malignant_misses"],
        ] for name, row in selected_round1],
        [1800, 1200, 1400, 1200, 1300, 1400, 1200],
    ))
    body.append(bullet("A 策略成本最低，但共同错误复核率为 0：两个模型一致但同时错误时无法触发复核。"))
    body.append(bullet("提高 C 阈值能发现更多共同错误，但自动处理率快速下降；C=0.80 时约 80% 样本进入复核。"))

    body.append(page_break())
    body.append(heading("3. 第二轮：Risk–Coverage 与 AURC"))
    body.append(rich_paragraph([
        ("共同积分区间：", True),
        ("Coverage 0–418/496（84.27%）。归一化 AURC 越低越好。", False),
    ]))
    body.append(table(
        ["策略", "Native 最大 Coverage", "Native AURC", "共同归一化 AURC"],
        [[
            row["strategy"] if row["strategy"] != "D" else f"D（λ={float(row['lambda']):.2f}）",
            pct(row["native_max_coverage"]),
            f"{float(row['native_raw_aurc']):.5f}",
            f"{float(row['common_normalized_aurc']):.5f}",
        ] for row in aurc_keep],
        [2700, 2200, 2200, 2500],
    ))
    body.append(image_paragraph("rIdRisk", "risk_coverage_round2", RISK_FIGURE, 6.45, 2))
    body.append(caption("图 2　B-V1、B-V2、C 与 D 的 Risk–Coverage 曲线；虚线为共同 AURC 上限 84.27%。"))
    body.append(paragraph("D（λ=0.05）的共同归一化 AURC 为 0.01200，较 B-V2 的 0.01397 低 0.00197（相对约 14.1%）；这是 validation 事后最优参数，不是已冻结规则。", bold=True, size=18, color="145B6B", after=100))

    body.append(heading("4. D 策略：同 Coverage 对比"))
    body.append(paragraph("D 以 V2 为最终预测，可靠性分数为 V2_confidence − λ × disagreement；低分样本进入复核。", size=18, after=70))
    body.append(table(
        ["Coverage", "策略", "选择性风险", "复核率", "未复核恶性漏诊", "共同错误复核", "最终预测错误复核"],
        [[
            pct(row["actual_coverage"]),
            row["strategy"] if row["strategy"] != "D" else "D（λ=0.05）",
            pct(row["selective_risk"]),
            pct(row["review_rate"]),
            row["unreviewed_malignant_misses"],
            pct(row["joint_error_review_recall"]),
            pct(row["final_prediction_error_review_recall"]),
        ] for row in target_keep],
        [1200, 1450, 1350, 1150, 1500, 1400, 1650],
    ))
    body.append(bullet("相对 B-V2：D 在 60% Coverage 风险持平，70% 降低 0.86 个百分点，80% 升高 0.50 个百分点。"))
    body.append(bullet("“共同错误复核”以 25 个 V1/V2 都错误样本为分母；“最终预测错误复核”以最终预测模型自身的错误为分母，两者不可混用。"))

    body.append(heading("5. 分歧与共同错误"))
    body.append(heading("5.1 预测分歧 78 例", level=2))
    reliability_rows = []
    name_map = {
        "always_V1_on_disagreement": "始终选 V1",
        "always_V2_on_disagreement": "始终选 V2",
        "choose_higher_raw_confidence": "选择原始置信度更高者",
        "delta_rank_auc_for_identifying_V2_correct": "置信度差识别 V2 正确的 AUC",
    }
    for row in disagreement:
        reliability_rows.append([
            name_map[row["rule"]],
            row["correct_count"] or "—",
            f"{float(row['accuracy']):.3f}",
        ])
    body.append(table(["规则 / 统计量", "正确数", "准确率 / AUC"], reliability_rows, [5200, 1800, 2400]))
    body.append(paragraph("始终选 V2 为 48/78（61.54%）；选择置信度更高模型仅提升到 49/78（62.82%）。未经跨模型校准的置信度差异不足以可靠决定分歧时相信谁。", size=18, after=90))

    body.append(heading("5.2 共同错误 25 例", level=2))
    body.append(table(
        ["C 阈值", "复核共同错误", "遗漏", "共同错误复核率", "遗漏恶性"],
        [[
            f"{float(row['c_threshold']):.2f}",
            row["joint_errors_reviewed"],
            row["joint_errors_missed"],
            pct(row["joint_error_review_recall"]),
            row["missed_malignant_count"],
        ] for row in common_detection],
        [1800, 2200, 1500, 2300, 1700],
    ))
    body.append(bullet("25 个共同错误的 disagreement 均为 0，因此 D 的 λ 惩罚对它们完全无效。"))
    body.append(bullet("6 个 V2 置信度 ≥0.80 的共同错误中，V1_conf<0.70 仅识别 2 个，V1_conf<0.74 识别 4 个，V1_conf<0.80 才识别全部。"))

    body.append(heading("6. 结论与限制"))
    body.append(bullet("V2 是明显优于 V1 的单模型基线；B-V2 的共同归一化 AURC 也明显优于 B-V1。"))
    body.append(bullet("D（λ=0.05）改善平均排序质量，但没有在关键 Coverage 上稳定优于 B-V2；不建议仅为 D 增加完整 V1 推理。"))
    body.append(bullet("若 V1 已因其他流程存在，D 可作为候选；仍需独立校准并在查看 test 前冻结 λ、Coverage 与复核规则。"))
    body.append(bullet("所有结论仅针对当前 validation；同一 validation 同时参与 checkpoint 选择与探索分析，不能宣称泛化提升。"))
    body.append(paragraph("未执行 test 评估。", bold=True, size=19, color="A33A2B", before=80, after=0))

    section = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="850" w:right="850" w:bottom="850" w:left="850" w:header="420" w:footer="420" w:gutter="0"/>'
        '<w:cols w:space="708"/><w:docGrid w:linePitch="312"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}">'
        "<w:body>" + "".join(body) + section + "</w:body></w:document>"
    )


def build_docx() -> None:
    required = [
        ROUND1 / "baselines.csv",
        ROUND1 / "aligned_v1_v2_scores.csv",
        ROUND1 / "threshold_sweep.csv",
        ROUND2 / "aurc_comparison.csv",
        ROUND2 / "target_coverage_comparison.csv",
        ROUND2 / "disagreement_reliability.csv",
        ROUND2 / "common_error_c_detection.csv",
        RISK_FIGURE,
        CONFIDENCE_FIGURE,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    document = build_document()
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rIdRisk" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/risk_coverage.png"/>
<Relationship Id="rIdConfidence" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/confidence_groups.png"/>
</Relationships>"""
    styles = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos" w:eastAsia="微软雅黑"/><w:lang w:val="zh-CN" w:eastAsia="zh-CN"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="90" w:line="280" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="145B6B"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="287687"/></w:rPr></w:style>
</w:styles>"""
    now = dt.datetime(2026, 11, 1, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=8))).isoformat()
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>Verify 实验报告（中文精简版）</dc:title><dc:subject>TN5000 validation-only Verify exploration</dc:subject><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion></Properties>"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        archive.write(RISK_FIGURE, "word/media/risk_coverage.png")
        archive.write(CONFIDENCE_FIGURE, "word/media/confidence_groups.png")

    print(OUTPUT)


if __name__ == "__main__":
    build_docx()
