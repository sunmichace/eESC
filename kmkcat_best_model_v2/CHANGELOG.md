# 变更日志

## 2026-04-04

### 本轮整理内容

- 为项目补齐并收口标准源码层，统一到 `src/eesc/`
- 为 `kcat_tc`、`km_tc`、`kcatkm_tc` 建立任务共享层
- 为任务入口补齐 `task-view` / `task-run`
- 将分析入口统一到 `eesc.cli`
- 删除旧的根目录分析脚本入口
- 删除 `scripts/` 下旧的分析薄封装
- 删除旧的 CLI 包装命令 `legacy-scripts` 和 `legacy-run`
- 更新根目录 README 和 `tasks/README.md`

### 当前保留的标准入口

- `env PYTHONPATH=src python -m eesc.cli tasks`
- `env PYTHONPATH=src python -m eesc.cli task-view <task>`
- `env PYTHONPATH=src python -m eesc.cli task-run <task> <alias>`
- `env PYTHONPATH=src python -m eesc.cli metrics`
- `env PYTHONPATH=src python -m eesc.cli training-r2`

### 说明

- 任务目录内部的 `legacy_shared.py` 仍然保留，它们是历史脚本到标准共享层的内部桥接，不再视为对外接口。
- 本轮没有迁移大体量实验资产，重点是规范入口、减少重复定义、提升可读性。
