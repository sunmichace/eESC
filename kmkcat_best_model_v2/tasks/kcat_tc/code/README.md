# kcat_tc/code

这个目录是 `kcat_tc` 任务的主实验工作区。和规范项目不同，这里把训练入口、原始 fold 数据、预处理特征、最佳权重和 stacking 结果都放在了一起。

为了阅读更清楚，可以按下面的角色理解：

## 目录分层

```text
code/
├── featers_train_fold_2_f_f.py       # 5-fold 训练入口
├── legacy_shared.py                  # 指向 src/eesc/tasks/kcat/ 的兼容层
├── tune_hyperparams_*.py             # 各模型的 Optuna 调参入口
├── auto_runner.py                    # 多次重复运行并保留最佳结果
├── kcat/                             # 当前任务的主数据目录
│   ├── fold_*.csv
│   └── preprocessed/*.npy
├── improved_best_model_fold*.pth     # 历史运行直接产出的权重
└── best_model_save/                  # 最佳模型与 stacking 相关的产物归档
```

## 哪些是“当前主目录”

- `kcat/`
  这是当前任务的主数据目录。训练脚本里常见的 `DATA_DIR = "./kcat/preprocessed/"` 指向的就是这里。

- `best_model_save/`
  这是产物归档目录，不是新的源码根。这里保存了最佳权重、训练日志、stacking 脚本和一份历史数据快照。

## 哪些文件是主要入口

- `featers_train_fold_2_f_f.py`
  直接启动 5-fold 训练。现在它主要负责训练循环和数据加载 orchestration，模型定义已经开始迁往标准源码层。

- `legacy_shared.py`
  这是从当前历史工作目录访问 `src/eesc/tasks/kcat/` 的兼容入口。`NpyDataset`、基础网络块和多种共享模型都通过它导出。

- `src/eesc/tasks/kcat/models.py`
  这是当前已经抽出的共享模型层。`CNNPredictor`、`ImprovedPredictorModel`、`GatedInteractionModel`、`DualTransformerFusionModel`、主训练使用的 `AttentionDTIModel`，以及调参使用的 `AttentionDTITuningModel` 都集中在这里。

- `src/eesc/tasks/kcat/stacking_runtime.py`
  这是 stacking 脚本共用的运行层。`best_model_save/essm_machine_xGboost_data*.py` 现在通过它复用 checkpoint 模型注册、评估函数和默认常量。

- `tune_hyperparams_CNNPredictor.py`
- `tune_hyperparams_AttentionDTIModel.py`
- `tune_hyperparams_DualTransformerFusionModel.py`
- `tune_hyperparams_ImprovedPredictorModel.py`
  这四个是当前最直接的调参入口。

- `best_model_save/essm_machine_xGboost_data.py`
- `best_model_save/essm_machine_xGboost_data_merge.py`
  这两个是 stacking 阶段的主入口。

## 关于 auto_runner.py

`auto_runner.py` 保留了实验期的自动重复运行逻辑，但文件里仍然写死了：

- `SCRIPT_NAME = "CNNPredictor.py"`
- `SOURCE_WEIGHT = "improved_best_model_fold3.pth"`

所以它更适合作为“历史自动化样板”，不是零配置即可直接运行的标准入口。

## 推荐用法

如果你从项目根目录操作，优先用统一 CLI 查看稳定别名：

```bash
env PYTHONPATH=src python -m eesc.cli task-view kcat
env PYTHONPATH=src python -m eesc.cli task-run kcat tune-cnn --dry-run
env PYTHONPATH=src python -m eesc.cli task-run kcat tune-cnn --show-template
```

如果你仍想按原始方式运行，就先进入这个目录再执行对应脚本。

如果你要改默认参数，优先编辑脚本顶部的常量区：

- 训练入口通常看 `ACTIVE_TRAIN_FOLDS`、`DEFAULT_EPOCHS`、`DEFAULT_BATCH_SIZE`
- 调参入口通常看 `TUNING_FOLD`、`TUNING_EPOCHS`、`TUNING_TRIALS`
- stacking 入口通常看 `src/eesc/tasks/kcat/stacking_runtime.py` 里的 `STACKING_FOLDS`、`BATCH_SIZE`、`DEFAULT_NUM_WORKERS`
