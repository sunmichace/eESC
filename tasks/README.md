# Tasks Overview

`tasks/` 保存三组实验任务目录。它们仍然保留原始实验期的组织方式，因此“代码、数据、权重、日志”是混放的，但对外入口已经统一到 `task-view` / `task-run`。

## 任务列表

| 目录 | 目标 | 当前状态 |
| --- | --- | --- |
| `kcat_tc/` | `kcat` 回归 | 训练入口、调参脚本、数据和最佳模型结果都比较完整。 |
| `km_tc/` | `Km` 回归 | 入口主要集中在 `code/best_model_save/`，但 stacking 共享层已经规范到 `src/eesc/tasks/km/`。 |
| `kcatkm_tc/` | 联合/扩展任务 | 结构接近 `kcat_tc/`，保留了完整 fold 数据和多模型结果。 |

## 通用结构

各任务目录通常包含一个 `code/` 子目录，内部常见内容如下：

- `featers_train_fold_2_f_f.py`：基础训练入口
- `tune_hyperparams_*.py`：针对不同模型的 Optuna 调参脚本
- `auto_runner.py`：重复运行并挑选最佳结果的自动执行脚本
- `<task_name>/`：fold 划分 CSV 和 `preprocessed/*.npy` 特征
- `best_model_save/`：训练日志、最佳权重、stacking 脚本和机器学习结果
- `legacy_shared.py` 或对应兼容层：把任务目录内部的历史脚本桥接到 `src/eesc/tasks/*/` 的标准共享模块

## 使用建议

- 运行训练脚本前，通常还是要进入对应任务的 `code/` 目录。
- 如果只是阅读结果，优先看 `best_model_save/` 和根目录 `reports/analysis_results/`。
- `km_tc/` 的目录组织和另外两个任务不完全一致，先看它自己的 README。
- `kcat_tc`、`km_tc` 和 `kcatkm_tc` 现在都已经有对应的标准共享层，只是完整度不同。
- 从项目根目录只保留统一 CLI，不再保留旧的包装命令。

```bash
env PYTHONPATH=src python -m eesc.cli task-view kcat
env PYTHONPATH=src python -m eesc.cli task-run kcat tune-cnn --dry-run
env PYTHONPATH=src python -m eesc.cli task-run kcat tune-cnn --show-template
env PYTHONPATH=src python -m eesc.cli task-view km
env PYTHONPATH=src python -m eesc.cli task-run km stacking-xgb --show-template
```

详细说明：

- [kcat_tc/README.md](/hdd/project/eESC/tasks/kcat_tc/README.md)
- [km_tc/README.md](/hdd/project/eESC/tasks/km_tc/README.md)
- [kcatkm_tc/README.md](/hdd/project/eESC/tasks/kcatkm_tc/README.md)
