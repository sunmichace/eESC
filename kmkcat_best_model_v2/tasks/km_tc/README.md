# km_tc

`km_tc` 对应 `Km` 回归任务。它仍然不像 `kcat_tc` / `kcatkm_tc` 那样保留完整的顶层训练链路，但现在已经补上一层标准共享结构，至少把当前可运行的 stacking 工作流整理成可读、可维护的形式。

## 目录结构

```text
tasks/km_tc/
└── code/
    └── best_model_save/
        ├── README.md
        ├── legacy_shared.py
        ├── km/
        │   ├── fold_*.csv
        │   └── preprocessed/*.npy
        ├── km_*_pth/
        ├── machine_results/
        ├── essm_machine_xGboost_data.py
        └── essm_machine_xGboost_data_merge_machine.py

src/eesc/tasks/km/
├── __init__.py
├── shared.py
├── models.py
└── stacking_runtime.py
```

## 当前状态

- `km_tc` 的可执行入口仍然主要集中在 `code/best_model_save/`。
- 两个 stacking 脚本已经改成薄封装，不再把 `Dataset`、基础 block、模型定义和评估函数全部塞在一个文件里。
- 共享实现现在统一放在 `src/eesc/tasks/km/`。
- 原始数据、权重、日志和输出文件路径保持不动，没有大规模搬迁实验资产。

## 共享层职责

- `src/eesc/tasks/km/shared.py`
  统一保存 `km` 分支的数据路径、`NpyDataset` 和基础网络块。

- `src/eesc/tasks/km/models.py`
  统一保存共享模型定义，并提供和历史 checkpoint 兼容的 `Stacking*` 包装类。

- `src/eesc/tasks/km/stacking_runtime.py`
  统一保存 stacking 默认参数、checkpoint 加载、评估函数和 `xgboost` 不可用时的 fallback。

- `code/best_model_save/legacy_shared.py`
  负责把历史工作目录桥接到新的 `src/eesc/tasks/km/` 共享层。

## 关键位置

- `code/best_model_save/essm_machine_xGboost_data.py`
  基础 stacking 入口，保留原始的 XGBoost-only 输出行为，生成 `machine_results/global_xgb_model.pkl` 和 `train_log.txt`。

- `code/best_model_save/essm_machine_xGboost_data_merge_machine.py`
  merge stacking 入口，保留 `XGB + Ridge + 最终融合` 行为，生成 `global_merge_xgb_model.pkl` 和 `global_merge_ridge_model.pkl`。

- `code/best_model_save/km_*_pth/`
  保存不同基础模型的最佳权重和训练日志。

- `code/best_model_save/km/preprocessed/`
  保存 stacking 使用的特征和标签文件，命名为 `fold_{fold}_{split}_{kind}.npy`。

- `code/best_model_save/machine_results/`
  保存 stacking 阶段输出的模型文件和日志。

## 建议使用方式

优先从项目根目录走统一 CLI：

```bash
env PYTHONPATH=src python -m eesc.cli task-view km
env PYTHONPATH=src python -m eesc.cli task-run km stacking-xgb --show-template
env PYTHONPATH=src python -m eesc.cli task-run km stacking-merge --dry-run
```

如果你要看当前默认参数，优先看：

1. `src/eesc/tasks/km/stacking_runtime.py`
2. `src/eesc/tasks/km/models.py`
3. `code/best_model_save/README.md`

## 备注

- 当前分支依然是“stacking 阶段最完整，训练阶段缺失较多”的状态，这一点没有被伪装成已经彻底统一。
- 这次整理的重点是把现有可运行部分规范化，而不是重新构造一套并不存在的训练入口。
