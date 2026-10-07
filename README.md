# TN5000：整图与人工 ROI 分类对照

本项目考察：向视觉语言模型额外提供人工标注 ROI，是否优于只提供整图完成甲状腺超声良恶性分类。

截至 2026-10-07，所有完成工作、结果与限制均汇总在本文件。原始 TN5000 图像、XML、模型权重、模型缓存和逐图运行产物均不进入 Git。

## 当前结论

固定 `Qwen/Qwen3-VL-4B-Instruct` 全量训练最佳 `checkpoint-epoch-2`，在去重且结构有效的固定 validation 集（496 张：良性 123、恶性 373）上：

| 输入 | Accuracy | Macro-F1 | 恶性敏感度 | 良性特异度 | 相对整图净正确数 |
|---|---:|---:|---:|---:|---:|
| V1：整图 | 85.28% | 0.79995 | 90.88% | 68.29% | 0 |
| O1：整图 + 原始人工 ROI | 83.87% | 0.77495 | 91.15% | 61.79% | -7 |
| D1：整图 + 同一张整图 | 80.24% | 0.75826 | 81.77% | 75.61% | -25 |
| R10：整图 + 每侧 10% 扩边 ROI | 83.67% | 0.77282 | 90.88% | 61.79% | -8 |

O1 纠正 10 例、改坏 17 例；没有观察到净收益。R10 纠正 8 例、改坏 16 例；扩边没有恢复 V1 表现。D1 说明重复提供同一整图本身也会改变模型输出，不能把 O1/R10 的变化完全归因于 ROI 像素。

这些都是同一个固定开发集上的描述性结果。checkpoint 由此 val 选择，不能当作最终泛化结论；test 尚未运行。没有患者或结节 ID，患者级独立性无法确认。

## 已实际完成的工作

### 数据与环境审计

- AutoDL：RTX 4090 D，24,564 MiB；Python 3.12.3；PyTorch 2.8.0+cu128。
- 全量检查 5,000 张 JPEG 和 XML：每张有一个已知标签对象，均可解码。
- XML `0=benign`、`1=malignant`；框为 `xmin, ymin, xmax, ymax`。
- ROI 坐标使用官方加载器规则：四个坐标均减 1，作为 Pillow 半开区间；不 padding、不夹紧、不修改原始文件。
- 119 个精确重复组，65 组跨官方 train/val/test。因此实验使用文件 SHA256 或解码 RGB SHA256 的连通重复组隔离，优先级 `test > val > train`。
- 已发现但未修复：`002589`（train，XML/JPEG 尺寸不一致）、`003813`（train，原始 xmax 超界 1 像素）、`004092`（test，XML/JPEG 尺寸不一致）。

官方图像级统计为 train 3,500、val 500、test 1,000。去重及结构检查后，本项目的 train 为 3,384（良性 999、恶性 2,385），固定 val 为 496（良性 123、恶性 373）。3,384 已经排除了 `002589` 和 `003813`，不能再次相减。

### 全量 V1 训练

基础模型为 `Qwen/Qwen3-VL-4B-Instruct`。LoRA：`r=4`、alpha=8、dropout=0、目标 `q_proj`/`v_proj`；AdamW，学习率 `1e-4`；自然类别比例，microbatch 1、梯度累积 2。

全量 train 实际完成 5 epoch、8,460 optimizer steps。按固定 val Macro-F1 早停，最佳为 epoch 2：Accuracy 0.852823、Macro-F1 0.799946，混淆矩阵 `[[84,39,0],[34,339,0]]`（行是真实 benign/malignant，列是预测 benign/malignant/parse_failed）。保留的 adapter SHA256：`e7541083da9725aa988408e431497007ce4e438117f8708c42d44a0f3a38b71c`。

### ROI 对照与诊断

所有 496 张固定 val 均使用同一 adapter、处理器、生成参数（`max_new_tokens=24`、`do_sample=false`）和解析规则。提示词不含真实标签、文件名、第一次回答或人工审计面板。O1、D1、R10 都将第二张图实际送入视觉编码器；整图第一张的张量与 V1 一致。

- O1：原始无 padding ROI。平均视觉 token 443.26，相对 V1 的 371.87 增加 19.2%；排除六例启动检查后平均单例耗时 160.42 ms，相对 V1 144.81 ms 增加 10.8%。
- D1：第二张为同一整图的像素完全相同副本。平均视觉 token 743.75；平均单例耗时 201.92 ms，相对同步复跑 V1 的 143.44 ms 增加 40.8%。
- R10：每侧扩展原框对应宽/高的 10%，左上向下取整、右下向上取整，再截到图像边界；其余 490 例未触边，6 例触边坐标保存在本地结果。平均视觉 token 447.69；平均单例耗时 159.64 ms，增加 11.3%。

三组均无解析失败、文本/视觉输入截断或输出长度截断。O1 的全部 27 个变化案例已做事后整理；其中出现的测量端点或界面标记只作描述，未用来改标注，也不能证明其导致预测变化。

## 冻结的最终测试规则

最终 test 尚未运行。届时 V1 与 ROI 两臂使用同一去重、结构有效清单；解析失败保留并计错。`004092` 从两臂主 test 同时排除：它的 XML 为 677x432，而 JPEG 为 718x500；虽转换后框在图像内，仍不能证明标注和图像对齐。不得缩放、夹紧或修复该框。冻结后的主 test 元数据清单为 991 张（良性 268、恶性 723）。

## 精简后的本地输出

`outputs/` 被 Git 忽略，仅保留后续工作有用的本地文件：

```text
outputs/
├── model_cache/             # Qwen 本地缓存；保留，换同一磁盘/GPU 不必重下
├── checkpoint-epoch-2/      # 选定 LoRA adapter；不提交 Git
└── final_val496/            # 配置、指标、逐例原始回答、O1 复核和扩边坐标
```

换到新实例通常需要重新下载模型；若携带 `outputs/model_cache/` 或挂载持久盘则可直接复用。逐图 PNG、早期冒烟、32 图拟合、256 图开发试验、非最佳 checkpoints、模型缓存以外的中间训练输出均已删除。

## 使用

数据不随仓库分发。下载后应放在：

```text
data/official/TN5000_forReview/
├── Annotations/<image_id>.xml
├── JPEGImages/<image_id>.jpg
└── ImageSets/Main/{train,val,test,trainval}.txt
```

安装最小审计依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
python scripts/check_tn5000.py --data-root data/official/TN5000_forReview --output outputs/phase1_audit --seed 20260923 --preview-count 20
```

`scripts/fetch_official.py` 可按官方元数据下载数据和基准代码；使用前请自行确认数据许可与磁盘空间。当前仓库只保留数据下载、环境检查和全量数据审计脚本。模型训练和诊断的完成记录、精简结果和已选 checkpoint 在本地 `outputs/` 中，模型权重与数据都不进入 Git。

## 项目目录

```text
.
├── README.md                # 唯一人类可读报告
├── results.json             # 关键机器可读结果
├── requirements.txt
├── evidence/                # 官方数据及源码定位元数据
├── scripts/                 # 下载、环境检查、数据审计
└── outputs/                 # 本地忽略：缓存、最佳 adapter、精简结果
```

没有新增训练、test 推理、重复 seed 或其他 ROI 比例搜索。
