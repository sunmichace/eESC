"""Dataset classes for feature extraction and DL training."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class EnzymeKineticsDataset(Dataset):
    """原始数据 → tokenized features (供特征提取使用)"""

    def __init__(self, dataframe, smiles_tokenizer, esm_tokenizer,
                 target_col, ph_scaler=None,
                 max_len_smiles=128, max_len_prot=1024):
        self.df = dataframe.reset_index(drop=True)
        self.smiles_tokenizer = smiles_tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.target_col = target_col
        self.max_len_smiles = max_len_smiles
        self.max_len_prot = max_len_prot

        required_cols = ['smiles', 'seq', 'ph', 'temperature', target_col]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Column {col} missing from dataframe")

        self.df['ph'] = pd.to_numeric(self.df['ph'], errors='coerce').fillna(7.0)
        self.df['temperature'] = pd.to_numeric(self.df['temperature'], errors='coerce').fillna(25.0)
        self.df[target_col] = pd.to_numeric(self.df[target_col], errors='coerce')
        self.df = self.df[self.df[target_col] > 0].reset_index(drop=True)

        self.labels = np.log10(self.df[target_col].values.astype(np.float32))

        features = self.df[['ph', 'temperature']].values.astype(np.float32)
        if ph_scaler is None:
            self.scaler = StandardScaler()
            self.normalized_features = self.scaler.fit_transform(features)
        else:
            self.scaler = ph_scaler
            self.normalized_features = self.scaler.transform(features)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        smiles = str(row['smiles'])
        smiles_enc = self.smiles_tokenizer(
            smiles, return_tensors="pt", padding='max_length',
            truncation=True, max_length=self.max_len_smiles,
            add_special_tokens=True
        )

        seq = str(row['seq'])
        if len(seq) > self.max_len_prot:
            seq = seq[:self.max_len_prot]
        prot_enc = self.esm_tokenizer(
            seq, return_tensors="pt", padding='max_length',
            truncation=True, max_length=self.max_len_prot
        )

        return {
            'smiles_ids': smiles_enc['input_ids'].squeeze(0),
            'smiles_mask': smiles_enc['attention_mask'].squeeze(0),
            'prot_ids': prot_enc['input_ids'].squeeze(0),
            'prot_mask': prot_enc['attention_mask'].squeeze(0),
            'extra_features': torch.tensor(self.normalized_features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class NpyDataset(Dataset):
    """预提取的 .npy 特征文件 → 训练/验证/测试"""

    def __init__(self, feat_path, label_path):
        self.features = np.load(feat_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'feat': torch.tensor(self.features[idx]),
            'label': torch.tensor(self.labels[idx])
        }
