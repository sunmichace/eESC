# kcatkm_tc/code/best_model_save

这个目录保存的是 `kcatkm_tc` 任务的“最佳模型产物层”，不是新的训练源码根。

## 目录角色

- `kcatkm_*_pth/`
  每个子目录对应一个模型分支，内部通常包含 `best.pth` 和 `training_history.log`。

- `machine_results/`
  stacking 阶段生成的 `pkl` 模型文件和 `train_log.txt` 汇总日志。

- `essm_machine_xGboost_data.py`
- `essm_machine_xGboost_data_merge.py`
  stacking 主脚本。现在它们已经不再内嵌基模型定义，而是复用 `src/eesc/tasks/kcatkm/stacking_runtime.py` 和 `src/eesc/tasks/kcatkm/models.py`。

- `kcatkm/`
  这是历史快照数据目录，更适合作为归档副本而不是主数据入口。

## 当前默认项

如果你要调整 stacking 的默认行为，优先看：

- [stacking_runtime.py](/hdd/project/eESC/src/eesc/tasks/kcatkm/stacking_runtime.py)
  这里统一定义了 `STACKING_FOLDS`、`BATCH_SIZE`、`DEFAULT_NUM_WORKERS`。

- [models.py](/hdd/project/eESC/src/eesc/tasks/kcatkm/models.py)
  这里统一定义了 `STACKING_MODEL_SPECS` 和 checkpoint 对应的共享模型包装类。
