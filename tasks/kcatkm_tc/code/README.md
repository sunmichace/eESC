# kcatkm_tc/code

这个目录是 `kcatkm_tc` 任务的主实验工作区。和规范项目不同，这里把训练入口、原始 fold 数据、预处理特征、最佳权重和 stacking 结果都放在了一起。

为了阅读更清楚，可以按下面的角色理解：

## 目录分层

```text
code/
├── featers_train_fold_2_f_f.py       # 5-fold 训练入口
├── legacy_shared.py                  # 指向 src/eesc/tasks/kcatkm/ 的兼容层
├── tune_hyperparams_*.py             # 各模型的 Optuna 调参入口
├── auto_runner.py                    # 多次重复运行并保留最佳结果
├── kcatkm/                           # 当前任务的主数据目录
│   ├── fold_*.csv
│   └── preprocessed/*.npy
├── improved_best_model_fold1.pth     # 历史运行直接产出的权重
└── best_model_save/                  # 最佳模型与 stacking 相关的产物归档
```

## 哪些文件是主要入口

- `featers_train_fold_2_f_f.py`
  直接启动 5-fold 训练。现在它主要负责训练循环和数据加载 orchestration，模型定义已经迁往标准源码层。

- `legacy_shared.py`
  这是从当前历史工作目录访问 `src/eesc/tasks/kcatkm/` 的兼容入口。

- `src/eesc/tasks/kcatkm/models.py`
  这是 `kcatkm` 的共享模型层。任务专属 stacking checkpoint 包装类和模型注册表定义在这里，而基础神经网络实现复用通用 `kcat` 模型层。

- `src/eesc/tasks/kcatkm/stacking_runtime.py`
  这是 stacking 脚本共用的运行层。`best_model_save/essm_machine_xGboost_data*.py` 现在通过它复用 checkpoint 模型注册、评估函数和默认常量。

- `tune_hyperparams_*.py`
  这些是当前最直接的调参入口。

## 推荐用法

如果你从项目根目录操作，优先用统一 CLI 查看稳定别名：

```bash
env PYTHONPATH=src python -m eesc.cli task-view kcatkm
env PYTHONPATH=src python -m eesc.cli task-run kcatkm tune-gated --dry-run
env PYTHONPATH=src python -m eesc.cli task-run kcatkm stacking-xgb --show-template
```

如果你仍想按原始方式运行，就先进入这个目录再执行对应脚本。

如果你要改默认参数，优先编辑这些常量区：

- 训练入口通常看 `ACTIVE_TRAIN_FOLDS`、`DEFAULT_EPOCHS`、`DEFAULT_BATCH_SIZE`
- 调参入口通常看 `TUNING_FOLD`、`TUNING_EPOCHS`、`TUNING_TRIALS`
- stacking 入口通常看 `src/eesc/tasks/kcatkm/stacking_runtime.py` 里的 `STACKING_FOLDS`、`BATCH_SIZE`、`DEFAULT_NUM_WORKERS`
