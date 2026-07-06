# Best Model - Enzyme Kinetics Prediction Pipeline

基于 **ESM2 + BARTSmiles** 特征提取和 **深度学习 + 机器学习 Stacking** 的酶动力学参数（km, kcat, kcat/Km）预测管线。

## 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Feature Extraction                        │
│  SMILES ─→ BARTSmiles ─→ 1024-dim                           │
│  Sequence ─→ ESM2-t33 ─→ 1280-dim                          │
│  pH + Temperature ─→ StandardScaler ─→ 2-dim               │
│  └───────────── Concatenate: [1024|1280|2] = 2306-dim ──→  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Deep Learning (5-Fold × 5 Models)              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. ImprovedPredictorModel                             │   │
│  │ 2. GatedInteractionModel                              │   │
│  │ 3. AttentionDTIModel                                  │   │
│  │ 4. DualTransformerFusionModel                         │   │
│  │ 5. CNNPredictor                                       │   │
│  └──────────────────────────────────────────────────────┘   │
│  每个模型在 5 折交叉验证上训练，保存最佳权重 (.pth)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           ML Stacking (元模型训练)                            │
│  输入: 5 个 DL 模型的预测值 + mean + std = 7 维元特征        │
│  注: 不再使用 pH/Temperature 作为 ML 输入（已冗余）           │
│                                                             │
│  km / kcat:     XGBoost + Ridge + Ensemble                  │
│  kcat/km:       Ridge (only)                                 │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
best_model/
├── shared/                              # 共享模块（三个任务共用同一套代码）
│   ├── dataset.py                       # 数据集类
│   ├── models.py                        # 5 种 DL 模型 + 辅助模块
│   ├── metrics.py                       # 评价指标工具
│   ├── extract_features.py              # 特征提取管线
│   ├── train_dl.py                      # DL 5 折训练管线
│   └── train_ml.py                      # ML Stacking 管线
├── configs/                             # 任务配置
│   ├── km_config.py
│   ├── kcat_config.py
│   └── kcatkm_config.py
├── run_km_pipeline.py                   # km 入口
├── run_kcat_pipeline.py                 # kcat 入口
├── run_kcatkm_pipeline.py               # kcatkm 入口
│
├── km_tc/                               # (原始代码，保留)
├── kcat_tc/                             # (原始代码，保留)
└── kcatkm_tc/                           # (原始代码，保留)
```

## Configuration

每个任务在 `configs/` 中通过参数进行配置，主要差异：

| 参数 | km | kcat | kcatkm |
|------|----|------|--------|
| 目标列 | km | kcat | kcatkm |
| 数据目录 | ./km/ | ./kcat/ | ./kcatkm/ |
| 特征目录 | ./km/preprocessed/ | ./kcat/preprocessed_esm2/ | ./kcatkm/preprocessed/ |
| 文件名前缀 | (无) | (无) | kcatkm |
| GPU | cuda:5 | cuda:5 | cuda |
| XGBoost | ✓ | ✓ | ✗ |
| Ridge | ✓ | ✓ | ✓ |

## Usage

所有命令在 `best_model/` 目录下执行：

```bash
# 运行完整管线（特征提取 → DL 训练 → ML Stacking）
python run_km_pipeline.py
python run_kcat_pipeline.py
python run_kcatkm_pipeline.py

# 仅运行特定步骤
python run_km_pipeline.py --extract     # 仅特征提取
python run_km_pipeline.py --train-dl    # 仅 DL 训练
python run_km_pipeline.py --train-ml    # 仅 ML Stacking
```

## Dependencies

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- XGBoost
- numpy, pandas, tqdm, joblib
