# 文件指南

建议交给 ChatGPT 讲解时，按 `README.md` → `docs/phase1_report.md` → `reports/phase1_summary.json` → `scripts/check_tn5000.py` 的顺序阅读。这样先理解研究问题和已知结论，再进入实现细节。

| 文件路径 | 文件用途 | 输入 | 输出 | 是否已在 AutoDL 成功运行 | 重点看什么 |
|---|---|---|---|---|---|
| `README.md` | 项目入口、运行方式和 Git 数据边界 | 无 | 人工可读说明 | 不适用，文档 | 当前阶段、复现命令、哪些内容不能上传 |
| `AGENTS.md` | 约束后续编码助手的操作与报告方式 | 当前项目约束 | 工作规则 | 不适用，文档 | 不上传数据/权重、不改原数据、不把标签写进提示 |
| `configs/phase1_data_audit.yaml` | 记录研究问题、相对路径、seed 和预览数量 | 无；目前不被脚本读取 | 配置记录 | 代码已经写好但尚未作为程序输入运行 | 它是审阅用元数据，CLI 参数才是实际运行输入 |
| `requirements.txt` | 第一阶段数据检查的最小依赖 | Python 环境 | Pillow、NumPy 依赖声明 | 这些版本已在 AutoDL 使用；本文件未重新安装验证 | 不含 Transformers、Qwen 或训练依赖 |
| `scripts/fetch_official.py` | 下载官方数据、基准代码和固定 GitHub 快照，校验 MD5 后解压 | `evidence/figshare_metadata.json`、`evidence/github_commit.json`、网络 | `downloads/`、`data/official/`、`vendor/` | AutoDL 实际运行成功 | 不覆盖现有解压目录；会下载数据但不会下载模型 |
| `scripts/check_environment.py` | 记录 Python、包、磁盘、nvidia-smi、nvcc，并运行微小 CUDA 张量测试 | 当前机器环境 | 本地 `evidence/environment.json` 和终端 JSON | AutoDL 实际运行成功 | 驱动、runtime、toolkit 是不同版本字段；输出含本机路径所以不进 Git |
| `scripts/check_tn5000.py` | 全量检查图像、XML、标签、框、split、重复图像，并生成固定种子预览 | `--data-root`、`--output`、`--seed`、`--preview-count` | summary、issues、manifest、duplicate groups、ROI 和联系表 | AutoDL 实际运行成功 | 脚本只读数据、拒绝覆盖输出；框转换和预览抽样规则 |
| `scripts/verify_audit.py` | 用官方解析函数和像素比较独立复核本轮结果 | 本地原始 ZIP、数据、官方代码快照和完整报告 | `verification.json` | AutoDL 实际运行成功 | 默认路径对应本轮 AutoDL 布局；公共克隆缺少本地文件时不能直接运行 |
| `docs/phase1_report.md` | 第一阶段中文主报告 | 已验证的环境与数据检查结果 | 面向人工审阅的结论 | 不适用，依据实际结果整理 | 三种状态、异常、信息缺口、下一阶段决策 |
| `docs/preview_guide.md` | 解释本地 ROI 联系表和统计图状态 | 本地 20 组预览 | 预览阅读说明 | 不适用，文档 | 预览能证明对应关系，不能证明模型收益或医学正确性 |
| `reports/environment_summary.json` | 可公开、脱敏的环境摘要 | 本轮环境检查结果 | 机器可读 JSON | 摘要由实际结果整理；不是一次新环境检测 | GPU、软件版本和未安装包 |
| `reports/phase1_summary.json` | 可公开的机器可读数据检查摘要 | 本轮最终 summary | JSON | 摘要由 AutoDL 成功结果整理 | 类别、split、异常、重复和验证状态 |
| `reports/phase1_issues.csv` | 公开的异常摘要 | 本轮 issues | CSV | 摘要由 AutoDL 成功结果整理 | 三个可定位标注疑点和两个聚合差异 |
| `reports/duplicate_groups.md` | 119 组精确重复图像及 split | 本轮重复哈希分组 | Markdown 表 | 分组检查在 AutoDL 实际运行成功 | 哪些重复组跨 train/val/test；没有自动删样本 |
| `evidence/figshare_metadata.json` | 官方 Figshare 文件名、下载 URL、大小和 MD5 | 官方 Figshare API | 下载脚本输入 | AutoDL 实际获取成功 | 数据版本和校验和，不含凭据 |
| `evidence/github_commit.json` | 固定官方仓库提交 | GitHub API | 下载脚本输入 | AutoDL 实际获取成功 | 固定 commit，避免代码随主分支变化 |
| `.gitignore` | 阻止数据、预览、模型、缓存、密钥和本机日志进入 Git | 文件路径规则 | Git 忽略行为 | 本次已用 `git check-ignore` 和候选清单检查 | `data/`、权重、预览、密钥规则是否生效 |
| `outputs/.gitkeep` | 保留空输出目录结构 | 无 | 空目录占位 | 不适用 | 实际输出默认被忽略 |

以下内容仅存在于当前 AutoDL，均不准备提交：

| 本地路径 | 分类 | 原因 |
|---|---|---|
| `data/` | TN5000 原始数据 | 数据集内容，禁止随代码上传 |
| `downloads/` | 原始下载包 | 大文件且可从官方源恢复 |
| `vendor/` | 官方代码快照 | 约 66 MB，可由固定 commit 恢复 |
| `reports/tn5000_seed*/` | 完整数据检查产物与 ROI 预览 | 包含数据派生图片、逐样本 manifest 和本机路径 |
| `reports/INSPECTION_REPORT.md` | 原始本机报告 | 含本机绝对路径，已由公开版报告替代 |
| `evidence/*.log`、`evidence/environment.json`、`evidence/github_tree.json` | 原始环境与运行日志 | 含本机路径或体积较大；公开摘要已脱敏 |
