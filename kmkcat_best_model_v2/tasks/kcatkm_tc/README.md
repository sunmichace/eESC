# kcatkm_tc

`kcatkm_tc` 是结构接近 `kcat_tc` 的另一组完整实验目录。从文件组织看，它同时保留了训练脚本、fold 数据、预处理特征、最佳权重和 stacking 结果。

## 目录结构

```text
tasks/kcatkm_tc/
└── code/
    ├── featers_train_fold_2_f_f.py
    ├── auto_runner.py
    ├── tune_hyperparams_*.py
    ├── improved_best_model_fold1.pth
    ├── kcatkm/
    │   ├── fold_*.csv
    │   └── preprocessed/*.npy
    └── best_model_save/
        ├── kcatkm_*_pth/
        ├── machine_results/
        └── essm_machine_xGboost_data*.py
```

## 关键文件

- `code/featers_train_fold_2_f_f.py`
  这是当前 5-fold 训练流程的主入口。现在它通过 `code/legacy_shared.py` 读取共享数据/模型定义，而不再独自保存所有模型实现。

- `code/legacy_shared.py`
  这是历史 `code/` 工作目录到标准源码层的兼容桥。`NpyDataset`、基础网络块和共享模型都从这里转发。

- `src/eesc/tasks/kcatkm/`
  这是 `kcatkm_tc` 的标准共享层，里面包含：
  `shared.py`、`models.py`、`stacking_runtime.py`。
  其中任务专属常量和 stacking checkpoint 包装定义在这里，而基础神经网络实现复用通用 `kcat` 模型层。

- `code/tune_hyperparams_*.py`
  覆盖 `CNNPredictor`、`AttentionDTIModel`、`DualTransformerFusionModel`、`ImprovedPredictorModel`、`GatedInteractionModel` 等模型的调参脚本。

- `code/auto_runner.py`
  用于多次重复训练并记录最佳结果。

- `code/kcatkm/`
  保存 fold CSV 和预处理 `.npy` 特征。

- `code/best_model_save/`
  保存最佳权重、训练历史、stacking 脚本与机器学习结果。

## 使用方式

推荐从任务目录内部运行：

```bash
cd /hdd/project/eESC/tasks/kcatkm_tc/code
python tune_hyperparams_GatedInteractionModel.py
```

如果只关心结果，重点看：

- `code/best_model_save/kcatkm_*_pth/training_history.log`
- `code/best_model_save/machine_results/train_log.txt`

如果你想从项目根目录以更规范的方式调用这条任务线，推荐：

```bash
env PYTHONPATH=src python -m eesc.cli task-view kcatkm
env PYTHONPATH=src python -m eesc.cli task-run kcatkm tune-gated --dry-run
env PYTHONPATH=src python -m eesc.cli task-run kcatkm stacking-xgb --show-template
```

额外的目录说明：

- [code/README.md](/hdd/project/eESC/tasks/kcatkm_tc/code/README.md)
- [code/best_model_save/README.md](/hdd/project/eESC/tasks/kcatkm_tc/code/best_model_save/README.md)
- [code/kcatkm/README.md](/hdd/project/eESC/tasks/kcatkm_tc/code/kcatkm/README.md)

## 备注

- 训练/调参脚本现在把主要默认项提到文件顶部常量，优先编辑这些名字：
  `ACTIVE_TRAIN_FOLDS`、`DEFAULT_EPOCHS`、`DEFAULT_BATCH_SIZE`、`TUNING_FOLD`、`TUNING_EPOCHS`、`TUNING_TRIALS`。
- stacking 脚本现在把共用默认项提到 [stacking_runtime.py](/hdd/project/eESC/src/eesc/tasks/kcatkm/stacking_runtime.py)，优先编辑：
  `STACKING_FOLDS`、`BATCH_SIZE`、`DEFAULT_NUM_WORKERS`。
- 当前目录命名与结构仍保留实验现场风格，但入口脚本现在已经接到标准共享层。
