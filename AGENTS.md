# Repository working agreement

## Project scope

This repository studies whole-image, manual-ROI-only, and whole-plus-ROI thyroid ultrasound classification on TN5000, followed by validation-only selective-review exploration.

## Current truth

- `README.md` is the public human-readable project record.
- `results.json` is the public machine-readable aggregate summary.
- `docs/figures/` contains aggregate plots only; sample-level artifacts remain ignored.
- Local runs under `outputs/` and original data under `data/` are intentionally ignored.
- The official paper defines XML label `0` as benign and `1` as malignant, and boxes as `xmin, ymin, xmax, ymax`.

## Safety rules

- Never add original TN5000 images, annotations, archives, ROI previews, patient data, model weights, caches, tokens, credentials, or SSH files to Git.
- Do not move, delete, relabel, clamp, or otherwise repair original data in place.
- Do not put ground-truth labels into inference prompts.
- Do not download models or start training/inference unless the user explicitly authorizes the next phase.
- Keep official-split and any deduplicated evaluation results separate and clearly named.
- Treat image IDs as image IDs only; no patient or nodule identifiers are available.

## Reporting language

Every report must distinguish:

1. AutoDL 实际运行成功 / actually run successfully on AutoDL.
2. 代码已经写好但尚未运行 / code written but not run.
3. 目前没有足够信息确认 / insufficient information to confirm.

List affected sample IDs for data anomalies. Never silently remove or correct them.

## Before proposing a commit

- Run syntax checks for repository-owned Python files.
- Inspect `git status --short --ignored` and the unignored file list.
- Scan unignored text files for secrets and files larger than 10 MB.
- Leave `git push`, remote creation, and publication to an explicit user instruction.
