"""kcat 任务配置"""

TARGET_COL = "kcat"
DATA_DIR = "./kcat/"
PREPROCESSED_DIR = "./kcat/preprocessed_esm2/"
FOLD_RANGE = (3, 4)       # 只跑 fold 3
FOLD_PREFIX = ""

DEVICE = "cuda:5"

# 深度学习训练参数
DL_EPOCHS = 300
DL_BATCH_SIZE = 64
DL_FOLD_RANGE = (3, 4)

# ML stacking 配置: kcat 使用 XGBoost + Ridge
ML_USE_XGBOOST = True
ML_USE_RIDGE = True
ML_BATCH_SIZE = 64
ML_FOLD_RANGE = (3, 4)
