# km best_model_save

`tasks/km_tc/code/best_model_save/` 是 `km_tc` 当前真正可执行的工作目录。和 `kcat_tc` / `kcatkm_tc` 不同，这个分支的可运行入口主要集中在这里。

## 目录角色

- `km/preprocessed/`：stacking 读取的 `.npy` 特征和标签，文件名形如 `fold_5_val_feat.npy`。
- `km_*_pth/`：各基础模型的最佳 checkpoint 与训练日志。
- `machine_results/`：stacking 训练输出，包含 `global_xgb_model.pkl`、`global_merge_xgb_model.pkl`、`global_merge_ridge_model.pkl` 和 `train_log.txt`。
- `essm_machine_xGboost_data.py`：基础 stacking 入口，保持原始的 XGBoost-only 行为。
- `essm_machine_xGboost_data_merge_machine.py`：merge/fusion stacking 入口，保留 XGB + Ridge + 最终融合逻辑。
- `legacy_shared.py`：把这个历史工作目录桥接到 `src/eesc/tasks/km/` 的兼容层。

## 规范化后的共享层

下面这些定义已经不再散落在两个 stacking 脚本中：

- `src/eesc/tasks/km/shared.py`：数据路径、Dataset、基础网络块。
- `src/eesc/tasks/km/models.py`：共享模型定义，以及与旧 checkpoint 兼容的 `Stacking*` 包装类。
- `src/eesc/tasks/km/stacking_runtime.py`：stacking 默认参数、checkpoint 加载、评估函数和 XGBoost fallback。

## 修改默认行为时看哪里

- 改 fold 范围、batch size、worker 数：看 `src/eesc/tasks/km/stacking_runtime.py`。
- 改 stacking 使用的基模型和 checkpoint 兼容默认值：看 `src/eesc/tasks/km/models.py`。
- 改旧工作目录相对路径：看 `legacy_shared.py` 和 `src/eesc/tasks/km/shared.py`。

## 运行建议

优先从项目根目录走统一入口：

```bash
env PYTHONPATH=src python -m eesc.cli task-view km
env PYTHONPATH=src python -m eesc.cli task-run km stacking-xgb --show-template
env PYTHONPATH=src python -m eesc.cli task-run km stacking-merge --dry-run
```

如果必须直接运行旧脚本，也要在这个目录下执行，保留它依赖的相对路径语义。
