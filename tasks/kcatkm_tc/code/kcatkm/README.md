# kcatkm_tc/code/kcatkm

这个目录是 `kcatkm_tc` 的主数据目录。

## 内容

- `fold_*.csv`
  原始 5-fold 划分结果。

- `preprocessed/*.npy`
  训练、验证、测试直接使用的预处理特征和标签。

## 备注

- 当前训练脚本中的 `DATA_DIR = "./kcatkm/preprocessed/"` 指向的就是这里。
- `best_model_save/kcatkm/` 下还有一份高度相近的历史快照，更适合作为归档副本而不是主入口。
