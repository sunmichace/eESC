"""km 任务配置"""

TARGET_COL = "km"
DATA_DIR = "./km/"
PREPROCESSED_DIR = "./km/preprocessed/"
FOLD_RANGE = (5, 6)       # 只跑 fold 5
FOLD_PREFIX = ""           # 文件名无前缀

DEVICE = "cuda:6"           # GPU 设备号

# 深度学习训练参数
DL_EPOCHS = 300
DL_BATCH_SIZE = 64
DL_FOLD_RANGE = (5, 6)

# ML stacking 配置: km 使用 XGBoost + Ridge
ML_USE_XGBOOST = True
ML_USE_RIDGE = True
ML_BATCH_SIZE = 64
ML_FOLD_RANGE = (5, 6)
