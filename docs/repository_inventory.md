# 当前项目文件盘点

本盘点用于决定哪些内容适合进入 GitHub。整理过程没有移动、删除或修改 TN5000 原始数据。

| 分类 | 当前主要路径 | Git 处理 | 说明 |
|---|---|---|---|
| 源代码 | `scripts/*.py` | 候选提交 | 4 个第一阶段脚本；均曾在 AutoDL 成功运行，本次又完成语法检查 |
| 配置文件 | `configs/phase1_data_audit.yaml`、`requirements.txt`、`AGENTS.md` | 候选提交 | 不含认证信息；YAML 是审阅用实验元数据，当前脚本仍使用 CLI 参数 |
| 环境信息 | `reports/environment_summary.json` | 候选提交 | 从实际环境检查结果脱敏整理，不含用户名、主目录、令牌或 SSH 信息 |
| 原始环境日志 | `evidence/environment.json`、`evidence/*.log` | 忽略 | 含本机绝对路径或命令原始输出；由公开摘要替代 |
| 官方发布元数据 | `evidence/figshare_metadata.json`、`evidence/github_commit.json` | 候选提交 | 下载脚本所需的公开元数据，不含私有凭据 |
| 数据检查报告 | `docs/phase1_report.md`、`reports/phase1_summary.json`、`reports/phase1_issues.csv`、`reports/duplicate_groups.md` | 候选提交 | 仅文字、数字和样本 ID；没有原始图像 |
| 完整本机检查产物 | `reports/tn5000_seed*/`、`reports/INSPECTION_REPORT.md` | 忽略 | 含逐样本 manifest、哈希、本机路径和预览 |
| ROI 预览图 | `reports/tn5000_seed20260923_final/previews/`、`contact_sheet_*.jpg` | 忽略，等待许可确认 | 已生成并验证，但包含 TN5000 图像内容 |
| 临时文件 | `.ipynb_checkpoints/`、`*.log`、`__pycache__/`、`*.tmp` | 忽略 | 不删除，只是不进入版本控制 |
| 原始数据 | `data/` | 忽略 | 约 228 MB，含 TN5000 JPG/XML；禁止上传 |
| 下载压缩包 | `downloads/` | 忽略 | 约 218 MB，可从官方源恢复 |
| 官方代码快照 | `vendor/` | 忽略 | 约 66 MB，由固定 Git commit 恢复 |
| 模型权重 | 当前未发现 | 模式级忽略 | `*.pt`、`*.pth`、`*.bin`、`*.safetensors`、`*.ckpt` 等均被排除 |
| 模型与训练缓存 | 当前未发现 | 目录级忽略 | Hugging Face、Torch、W&B、MLflow、runs 等路径均被排除 |
| 后续输出 | `outputs/` | 只提交 `.gitkeep` | 运行内容默认不进入 Git |

## 敏感信息检查

在整理前和候选提交文件生成后都进行了模式扫描。没有发现：

- 密码、访问令牌、API key 或 Bearer token；
- RSA、OpenSSH 或 EC 私钥；
- `.env`、`.netrc`、`.npmrc`、`.pypirc`；
- SSH 密钥或云服务凭据文件。

原始日志中的 `/root/...` 是本机绝对路径，不是认证信息；为减少机器信息暴露，原始日志仍被忽略。官方提交元数据可能包含官方仓库作者公开信息，它来自公共 GitHub API，不是本机账号凭据。

## 大文件结论

超过 10 MB 的现有文件均位于已忽略目录，主要是官方数据 ZIP 和数据派生预览。候选提交集合中没有超过 10 MB 的文件，也没有模型权重。
