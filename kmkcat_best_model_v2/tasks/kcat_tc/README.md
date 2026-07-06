# kcat_tc

`kcat_tc` 是当前目录里结构最完整的单任务分支，用于 `kcat` 回归实验。

## 目录结构

```text
tasks/kcat_tc/
└── code/
    ├── featers_train_fold_2_f_f.py
    ├── auto_runner.py
    ├── tune_hyperparams_*.py
    ├── improved_best_model_fold*.pth
    ├── kcat/
    │   ├── fold_*.csv
    │   └── preprocessed/*.npy
    └── best_model_save/
        ├── kcat_*_pth/
        ├── machine_results/
        └── essm_machine_xGboost_data*.py
```

## 关键文件

- `code/featers_train_fold_2_f_f.py`
  这是当前 5-fold 训练流程的主入口。现在它通过 `code/legacy_shared.py` 读取共享数据/模型定义，而不再独自保存所有模型实现。

- `code/legacy_shared.py`
  这是历史 `code/` 工作目录到标准源码层的兼容桥。`NpyDataset`、基础网络块和一部分共享模型都从这里转发。

- `src/eesc/tasks/kcat/models.py`
  这是当前已经抽出的 `kcat` 共享模型模块。`CNNPredictor`、`ImprovedPredictorModel`、`GatedInteractionModel`、`DualTransformerFusionModel`、主训练用的 `AttentionDTIModel`，以及调参用的 `AttentionDTITuningModel` 都已集中到这里。

- `src/eesc/tasks/kcat/stacking_runtime.py`
  这是当前 stacking 共用运行层，统一保存 checkpoint 模型注册、评估函数，以及 `STACKING_FOLDS` / `BATCH_SIZE` / `DEFAULT_NUM_WORKERS` 等默认项。

- `code/tune_hyperparams_*.py`
  分别对应不同模型的 Optuna 调参脚本。常见模型包括 `CNNPredictor`、`AttentionDTIModel`、`DualTransformerFusionModel`、`ImprovedPredictorModel`。

- `code/auto_runner.py`
  用于重复运行指定训练脚本，持续刷新最佳指标，并把最佳权重复制为 `best.pth`。

- `code/kcat/`
  保存 5-fold 划分 CSV 和 `.npy` 特征，是训练和验证时的核心输入目录。

- `code/best_model_save/`
  保存各模型的 `best.pth`、`training_history.log`、stacking 脚本和 `machine_results/` 汇总结果。

## 使用方式

推荐从 `code/` 目录启动：

```bash
cd /hdd/project/eESC/tasks/kcat_tc/code
python tune_hyperparams_CNNPredictor.py
```

如果你想从项目根目录以更规范的方式调用这条任务线，推荐：

```bash
env PYTHONPATH=src python -m eesc.cli task-view kcat
env PYTHONPATH=src python -m eesc.cli task-run kcat tune-cnn --dry-run
env PYTHONPATH=src python -m eesc.cli task-run kcat stacking-xgb --dry-run
env PYTHONPATH=src python -m eesc.cli task-run kcat tune-cnn --show-template
```

额外的目录说明：

- [code/README.md](/hdd/project/eESC/tasks/kcat_tc/code/README.md)
- [code/best_model_save/README.md](/hdd/project/eESC/tasks/kcat_tc/code/best_model_save/README.md)
- [code/kcat/README.md](/hdd/project/eESC/tasks/kcat_tc/code/kcat/README.md)

如果要查看重复训练日志，重点看：

- `code/best_model_save/kcat_*_pth/training_history.log`
- `code/best_model_save/machine_results/train_log.txt`

## 备注

- 脚本依赖相对路径，通常不能直接从项目根目录运行。
- 训练/调参脚本现在把主要默认项提到文件顶部常量，优先编辑这些名字：
  `ACTIVE_TRAIN_FOLDS`、`DEFAULT_EPOCHS`、`DEFAULT_BATCH_SIZE`、`TUNING_FOLD`、`TUNING_EPOCHS`、`TUNING_TRIALS`。
- stacking 脚本现在把共用默认项提到 [stacking_runtime.py](/hdd/project/eESC/src/eesc/tasks/kcat/stacking_runtime.py)，优先编辑：
  `STACKING_FOLDS`、`BATCH_SIZE`、`DEFAULT_NUM_WORKERS`。
- 文件名保留了实验期命名风格，建议把这里视为“原始实验现场”，而不是整理好的模块化代码。
- 这一轮新增的 `task-view` / `task-run` 是规范化包装层；其中 `kcat` 分支的一部分共享模型已经真正迁到 `src/eesc/tasks/kcat/`，但训练入口和参数配置仍主要留在原始脚本里。
