# kcat_tc/code/best_model_save

这个目录保存的是 `kcat_tc` 任务的“最佳模型产物层”，不是新的训练源码根。

## 目录角色

- `kcat_*_pth/`
  每个子目录对应一个模型分支，内部通常包含：
  - `best.pth`
  - `training_history.log`
  - 与该最佳模型一起保留的模型源码

- `machine_results/`
  stacking 阶段生成的 `pkl` 模型文件和 `train_log.txt` 汇总日志。

- `essm_machine_xGboost_data.py`
- `essm_machine_xGboost_data_merge.py`
  stacking 主脚本。现在它们已经不再内嵌基模型定义，而是复用 `src/eesc/tasks/kcat/stacking_runtime.py` 和 `src/eesc/tasks/kcat/models.py`。

- `kcat/`
  这是历史快照数据目录，和上一级 `../kcat/` 内容高度相近，更适合视为归档副本而不是主数据入口。

## 使用判断

- 如果你的目标是“重新训练或重新调参”，优先回到上一级 `code/`。
- 如果你的目标是“分析最佳模型结果或继续做 stacking”，这里才是主要工作区。

## 读取顺序

建议按下面顺序阅读：

1. `machine_results/train_log.txt`
2. 各模型目录下的 `training_history.log`
3. `essm_machine_xGboost_data.py`

## 当前默认项

如果你要调整 stacking 的默认行为，优先看：

- [stacking_runtime.py](/hdd/project/eESC/src/eesc/tasks/kcat/stacking_runtime.py)
  这里统一定义了 `STACKING_FOLDS`、`BATCH_SIZE`、`DEFAULT_NUM_WORKERS`。

- [models.py](/hdd/project/eESC/src/eesc/tasks/kcat/models.py)
  这里统一定义了 `STACKING_MODEL_SPECS` 和 checkpoint 对应的共享模型包装类。
