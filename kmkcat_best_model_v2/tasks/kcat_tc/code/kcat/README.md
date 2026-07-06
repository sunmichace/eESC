# kcat_tc/code/kcat

这是 `kcat_tc` 任务的主数据目录。

## 内容

- `fold_*.csv`
  5-fold 训练、验证、测试划分文件。

- `preprocessed/*.npy`
  预处理后的特征和标签数组，是训练脚本真正读取的输入。

## 命名规则

文件名通常遵循：

```text
kcat_fold_<fold_id>_<split>_<kind>.npy
```

例如：

- `kcat_fold_1_train_feat.npy`
- `kcat_fold_1_train_label.npy`
- `kcat_fold_1_val_feat.npy`
- `kcat_fold_1_test_label.npy`

## 使用说明

对于 `kcat_tc` 这条任务线，训练脚本里默认的 `DATA_DIR` 指向的是这个目录下的 `preprocessed/`，因此这里应视为当前主数据入口。
