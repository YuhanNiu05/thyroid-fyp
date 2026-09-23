# TN5000 整图与人工 ROI 对照实验

本项目计划在 TN5000 上验证：向视觉语言模型同时提供整张甲状腺超声图像与人工标注 ROI，相比只提供整图，是否能改善良恶性分类表现。计划主模型为 `Qwen/Qwen3-VL-4B-Instruct`。

当前只完成第一阶段的环境检查和数据检查。尚未下载模型，尚未进行训练、推理、强化学习或界面开发。第一阶段发现官方划分中存在跨集合的完全相同图像，因此在开始正式实验前必须先确定重复样本处理方案。

## 当前状态

| 工作 | 状态 |
|---|---|
| AutoDL GPU、Python、PyTorch、CUDA、磁盘检查 | AutoDL 实际运行成功 |
| TN5000 5,000 张图像与 XML 全量检查 | AutoDL 实际运行成功 |
| 固定种子生成 20 个整图、框和 ROI 预览 | AutoDL 实际运行成功；预览暂不进入 Git |
| 检查脚本与独立验证脚本 | AutoDL 实际运行成功 |
| Qwen 环境、模型加载、推理和训练 | 尚未验证 |
| 患者级或结节级数据隔离 | 目前没有足够信息确认 |

完整结果见 [第一阶段报告](docs/phase1_report.md)，文件阅读顺序见 [文件指南](docs/file_guide.md)。

## 目录说明

```text
.
├── README.md
├── AGENTS.md
├── configs/                  # 不含密码的实验元数据与默认值
├── docs/                     # 面向人工审阅和 ChatGPT 讲解的文档
├── evidence/                 # 可公开的官方发布元数据；本机日志被忽略
├── outputs/                  # 后续运行输出，内容默认不进 Git
├── reports/                  # 脱敏、汇总后的检查结果
├── scripts/                  # 下载、环境检查、数据审计和复核脚本
└── requirements.txt          # 第一阶段数据检查的最小 Python 依赖
```

本机还存在 `data/`、`downloads/`、`vendor/`、完整运行报告和 ROI 预览。这些内容保留原位，但被 `.gitignore` 排除。

## 安装第一阶段检查依赖

以下命令只安装数据检查所需的 Pillow 和 NumPy，不安装 Qwen、Transformers 或训练框架：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`scripts/check_environment.py` 也会检查 PyTorch 和 CUDA；若环境中没有 PyTorch，它会如实记录缺失。AutoDL 实测环境使用 Python 3.12.3、PyTorch 2.8.0+cu128 和 Pillow 11.3.0。这里不把当前机器环境误写成普适安装方案。

## 准备数据

TN5000 数据不随本仓库分发。请从[官方 Figshare 页面](https://doi.org/10.6084/m9.figshare.28455641)取得数据，并保持以下结构：

```text
data/official/TN5000_forReview/
├── Annotations/<image_id>.xml
├── JPEGImages/<image_id>.jpg
└── ImageSets/Main/{train,val,test,trainval}.txt
```

仓库中保留了官方发布元数据，`scripts/fetch_official.py` 可以复现本轮下载和校验过程。该脚本会下载约 230 MB 的数据与代码，只应在确认数据许可和存储位置后主动运行。它不会下载模型。

## 运行数据检查

先记录环境：

```bash
python scripts/check_environment.py
```

再使用一个全新的输出目录运行全量检查。脚本拒绝覆盖已有输出：

```bash
python scripts/check_tn5000.py \
  --data-root data/official/TN5000_forReview \
  --output outputs/phase1_audit \
  --seed 20260923 \
  --preview-count 20
```

本轮在 AutoDL 上实际使用的输出目录是 `reports/tn5000_seed20260923_final/`，该目录含数据派生文件和预览图，因此不进入 Git。公开汇总位于：

- [环境摘要](reports/environment_summary.json)
- [数据检查摘要](reports/phase1_summary.json)
- [异常摘要](reports/phase1_issues.csv)
- [重复图像分组](reports/duplicate_groups.md)

`scripts/verify_audit.py` 是本轮的独立复核脚本，实际验证过官方加载器坐标解析、20 个 ROI 像素一致性和官方 ZIP 文件一致性。它依赖本地官方源码快照和完整运行产物，默认路径也是本轮 AutoDL 路径布局；克隆后的公共仓库不能在缺少这些本地文件时直接运行。

## 下一步实验计划

1. 人工确认 20 个 ROI 的裁剪语义和视觉质量。
2. 确定 119 组精确重复图像、65 组跨划分重复以及 3 个标注疑点的处理协议，并保留官方划分作为可比基线。
3. 在相同样本、标签、随机种子和评估指标下构建“仅整图”与“整图＋人工 ROI”两组输入。
4. 再安装和验证 `Qwen/Qwen3-VL-4B-Instruct`，先做单样本显存与输入格式检查，然后才开始正式训练或推理。
5. 使用宏平均 F1、逐类召回等适合类别不均衡的指标，并做配对比较。真实标签只用于监督和评估，不写进推理提示。

## Git 数据边界

TN5000 原图、XML、下载压缩包、ROI 预览、模型权重、模型缓存、训练输出、密钥和本机认证配置均不进入 Git。20 个预览已经在本地生成并验证，但需要先确认数据集许可再决定是否单独发布。提交前建议运行：

```bash
git status --short --ignored
git ls-files --others --exclude-standard
```
