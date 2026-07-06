"""kcatkm 任务配置"""

TARGET_COL = "kcatkm"
DATA_DIR = "./kcatkm/"
PREPROCESSED_DIR = "./kcatkm/preprocessed/"
FOLD_RANGE = (1, 2)       # 只跑 fold 1
FOLD_PREFIX = "kcatkm"     # 文件名前缀

DEVICE = "cuda"

# 深度学习训练参数
DL_EPOCHS = 300
DL_BATCH_SIZE = 64
DL_FOLD_RANGE = (1, 2)

# ML stacking 配置: kcatkm 仅使用 Ridge (原始代码中 XGBoost 被注释掉)
ML_USE_XGBOOST = False
ML_USE_RIDGE = True
ML_BATCH_SIZE = 64
ML_FOLD_RANGE = (1, 2)
