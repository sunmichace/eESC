# eESC

这是一个以实验资产为主的酶动力学回归项目。仓库里保留了训练脚本、模型权重、fold 数据、stacking 结果和分析图表，因此它不是一个从零设计的轻量工程，而是在保留原始实验目录的前提下做规范化整理。

## 当前状态

项目已经完成第一阶段规范化：

- 统一源码入口位于 `src/eesc/`
- 统一命令行入口位于 `python -m eesc.cli`
- 统一任务入口通过 `task-view` / `task-run` 暴露
- `kcat_tc`、`km_tc`、`kcatkm_tc` 都已经接入标准共享层
- 旧的根目录脚本入口和旧的 CLI 包装命令已经移除

当前还没有彻底重构完成：

- 数据、代码、日志、权重仍然混放
- 历史命名仍然存在，例如 `featers_train_fold_2_f_f.py`
- `km_tc` 目前只规范了 stacking 入口，没有补齐完整训练链路

## 这次整理做了什么

### 1. 建立标准源码层

已经整理出：

- `src/eesc/project.py`
- `src/eesc/cli.py`
- `src/eesc/task_views.py`
- `src/eesc/analysis/`
- `src/eesc/tasks/`

其中：

- `project.py` 负责统一项目路径和任务元数据
- `cli.py` 提供统一命令入口
- `task_views.py` 提供任务视图和稳定别名
- `analysis/` 提供指标图和训练曲线的分析逻辑
- `tasks/` 承接各任务共享代码

### 2. 建立任务共享层

已经补齐：

- `src/eesc/tasks/kcat/`
- `src/eesc/tasks/km/`
- `src/eesc/tasks/kcatkm/`

这些模块现在承接：

- 数据路径和 `Dataset`
- 基础网络 block
- 共享模型定义
- stacking 默认参数
- checkpoint 加载逻辑
- 评估函数和元模型辅助函数

这样历史脚本本身更多只负责 orchestration，可读性比原先高很多。

### 3. 删除旧对外接口

已经删除：

- 根目录旧分析脚本入口
- `scripts/` 下旧分析薄封装
- `legacy-scripts`
- `legacy-run`

现在从项目根目录只保留新的标准入口。

### 4. 保留内部兼容层

以下兼容 shim 仍然保留，因为任务目录内部的历史脚本还依赖它们：

- `tasks/kcat_tc/code/legacy_shared.py`
- `tasks/km_tc/code/best_model_save/legacy_shared.py`
- `tasks/kcatkm_tc/code/legacy_shared.py`

这些文件不是对外接口，而是内部桥接层。

## 目录结构

```text
eESC/
├── pyproject.toml
├── README.md
├── src/
│   └── eesc/
│       ├── cli.py
│       ├── project.py
│       ├── task_views.py
│       ├── analysis/
│       └── tasks/
│           ├── kcat/
│           ├── km/
│           └── kcatkm/
├── reports/
│   └── analysis_results/
└── tasks/
    ├── README.md
    ├── kcat_tc/
    ├── km_tc/
    └── kcatkm_tc/
```

## 三个任务的状态

| 任务 | 当前状态 |
| --- | --- |
| `kcat_tc` | 最完整，训练、调参、stacking 都已有规范化共享层。 |
| `km_tc` | 主要规范了 `best_model_save/` 下两个 stacking 入口，共享层已抽到 `src/eesc/tasks/km/`。 |
| `kcatkm_tc` | 结构接近 `kcat_tc`，训练、调参、stacking 都已接入共享层。 |

## 如何使用

### 1. 查看项目入口

```bash
cd /hdd/project/eESC
env PYTHONPATH=src python -m eesc.cli tasks
env PYTHONPATH=src python -m eesc.cli task-view kcat
env PYTHONPATH=src python -m eesc.cli task-view km
env PYTHONPATH=src python -m eesc.cli task-view kcatkm
```

如果你只是接手项目，先执行这几条。

### 2. 生成分析图

生成 stacking 指标图：

```bash
cd /hdd/project/eESC
env PYTHONPATH=src python -m eesc.cli metrics
```

生成重复训练 R2 曲线：

```bash
cd /hdd/project/eESC
env PYTHONPATH=src python -m eesc.cli training-r2
```

### 3. 运行任务入口

从根目录统一通过 `task-run` 调用：

```bash
env PYTHONPATH=src python -m eesc.cli task-run kcat train-5fold --dry-run
env PYTHONPATH=src python -m eesc.cli task-run kcat tune-cnn --show-template
env PYTHONPATH=src python -m eesc.cli task-run kcat stacking-xgb --dry-run

env PYTHONPATH=src python -m eesc.cli task-run km stacking-xgb --show-template
env PYTHONPATH=src python -m eesc.cli task-run km stacking-merge --dry-run

env PYTHONPATH=src python -m eesc.cli task-run kcatkm train-5fold --dry-run
env PYTHONPATH=src python -m eesc.cli task-run kcatkm tune-cnn --show-template
```

`--show-template` 会告诉你：

- 实际映射到哪个底层脚本
- 当前默认参数应该去哪里改
- 推荐从哪个常量或模块入手

### 4. 如果已经安装为包

```bash
pip install -e .
eesc tasks
eesc task-view kcat
eesc task-run km stacking-xgb --show-template
```

## 默认参数应该改哪里

### `kcat_tc`

- 训练/调参默认项：直接看对应脚本顶部常量
- stacking 默认项：`src/eesc/tasks/kcat/stacking_runtime.py`
- 共享模型定义：`src/eesc/tasks/kcat/models.py`

### `km_tc`

- stacking 默认项：`src/eesc/tasks/km/stacking_runtime.py`
- stacking 基模型注册和 checkpoint 兼容默认值：`src/eesc/tasks/km/models.py`
- 目录内部桥接：`tasks/km_tc/code/best_model_save/legacy_shared.py`

### `kcatkm_tc`

- 训练/调参默认项：直接看对应脚本顶部常量
- stacking 默认项：`src/eesc/tasks/kcatkm/stacking_runtime.py`
- 共享模型定义：`src/eesc/tasks/kcatkm/models.py`

## 依赖

项目根依赖定义在 `pyproject.toml`。

只做结果分析时，基础依赖主要是：

- `numpy`
- `matplotlib`

如果要运行训练或 stacking 脚本，通常还需要：

- `torch`
- `scikit-learn`
- `tqdm`
- `optuna`
- `joblib`

部分 stacking 环境还可能需要额外安装 `xgboost`。

如果你要安装完整实验依赖，建议：

```bash
pip install -e ".[legacy-experiments]"
```

如果还要运行依赖 XGBoost 的分支，再额外安装：

```bash
pip install xgboost
```

## 已知限制

- 仓库仍然保留大量实验资产，不是轻量项目。
- 历史目录结构没有彻底重排，主要为了降低风险。
- `km_tc` 目前没有像 `kcat_tc` 那样恢复完整训练入口。
- 一些历史脚本仍然依赖相对路径和当前工作目录。
- 日志解析依赖既有日志格式，如果日志模板变化，分析脚本需要同步调整。

## 建议阅读顺序

1. 本 README
2. `tasks/README.md`
3. 对应任务目录下的 README
4. `src/eesc/task_views.py`
5. `src/eesc/tasks/<task>/`

## 后续建议

1. 继续把训练流程收口成统一标准入口，而不只是 `task-run` 包装。
2. 继续把历史脚本中的业务逻辑前移到 `src/eesc/tasks/*/`。
3. 为每个任务补最小可复现实验说明和依赖版本。
4. 逐步把代码、数据、产物拆层，而不是长期混放。
