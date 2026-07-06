"""
特征提取: BARTSmiles(1024) + ESM2-t33(1280) + env(2) → 2306-dim 特征.
仍保留 pH/Temperature 在特征中 (供深度学习模型使用), 但 ML 阶段不再使用.
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from shared.dataset import EnzymeKineticsDataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

SMI_DIM = 1024
PROT_DIM = 1280
ENV_DIM = 2
TOTAL_DIM = SMI_DIM + PROT_DIM + ENV_DIM  # 2306


def extract_features_for_dataset(smiles_encoder, prot_encoder, dataset, batch_size, device, desc="Extracting"):
    """拼接特征: [smiles(1024) | prot(1280) | env(2)] → (N, 2306)"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    smiles_encoder.eval()
    prot_encoder.eval()

    all_feats = []
    all_labels = []

    loop = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in loop:
            s_ids = batch['smiles_ids'].to(device)
            s_mask = batch['smiles_mask'].to(device)
            p_ids = batch['prot_ids'].to(device)
            p_mask = batch['prot_mask'].to(device)
            extra = batch['extra_features']
            labels = batch['labels']

            # BARTSmiles → 1024 (mean pooling)
            smiles_out = smiles_encoder(input_ids=s_ids, attention_mask=s_mask)
            mask_exp = s_mask.unsqueeze(-1).expand(smiles_out.last_hidden_state.size()).float()
            smiles_emb = torch.sum(
                smiles_out.last_hidden_state * mask_exp, 1
            ) / torch.clamp(mask_exp.sum(1), min=1e-9)

            # ESM2-t33 → 1280 ([CLS])
            prot_out = prot_encoder(input_ids=p_ids, attention_mask=p_mask)
            prot_emb = prot_out.last_hidden_state[:, 0, :]

            # 拼接 [smiles(1024) | prot(1280) | env(2)]
            combined = torch.cat([smiles_emb, prot_emb, extra.to(device)], dim=1)
            all_feats.append(combined.cpu().numpy())
            all_labels.append(labels.numpy())

    return {
        'feat': np.concatenate(all_feats, axis=0),
        'label': np.concatenate(all_labels, axis=0),
    }


def run_extraction(target_col, data_dir, save_dir, device, bartsmiles_path="BARTSmiles/",
                   esm2_path="facebook/esm2_t33_650M_UR50D", batch_size=16, fold_range=(1, 6)):
    """
    特征提取主入口.

    Args:
        target_col: 目标列名 (如 "km", "kcat", "kcatkm")
        data_dir: CSVs 所在目录
        save_dir: 特征保存目录
        device: torch device
        bartsmiles_path: BARTSmiles 模型路径
        esm2_path: ESM2 模型路径
        batch_size: 批大小
        fold_range: (start, end) fold 范围, 如 (1, 6) 表示 fold 1~5
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*10} Extract Features for Target: {target_col} {'='*10}")
    print(f"Feature layout: [smiles({SMI_DIM}) | prot({PROT_DIM}) | env({ENV_DIM})] = {TOTAL_DIM}")

    # 1. BARTSmiles
    print("Loading BARTSmiles tokenizer and model...")
    smiles_tokenizer = AutoTokenizer.from_pretrained(bartsmiles_path, add_prefix_space=True)
    if smiles_tokenizer.pad_token is None:
        smiles_tokenizer.pad_token = "<pad>"
    if smiles_tokenizer.bos_token is None:
        smiles_tokenizer.bos_token = "<s>"
    if smiles_tokenizer.eos_token is None:
        smiles_tokenizer.eos_token = "</s>"
    if smiles_tokenizer.unk_token is None:
        smiles_tokenizer.unk_token = "<unk>"

    smiles_encoder = AutoModel.from_pretrained(bartsmiles_path).to(device)
    smiles_encoder.eval()

    # 2. ESM2
    print("Loading ESM2-t33-650M tokenizer and model...")
    esm_tokenizer = AutoTokenizer.from_pretrained(esm2_path)
    prot_encoder = AutoModel.from_pretrained(esm2_path).to(device)
    prot_encoder.eval()

    # 3. 逐折提取
    for fold in range(fold_range[0], fold_range[1]):
        print(f"\n--- Fold {fold}/5 ---")

        train_file = os.path.join(data_dir, f"fold_{fold}_train.csv")
        val_file = os.path.join(data_dir, f"fold_{fold}_val.csv")
        test_file = os.path.join(data_dir, f"fold_{fold}_test.csv")

        try:
            df_train = pd.read_csv(train_file)
            df_val = pd.read_csv(val_file)
            df_test = pd.read_csv(test_file)
        except Exception as e:
            print(f"Error reading fold {fold}: {e}")
            continue

        train_dataset = EnzymeKineticsDataset(df_train, smiles_tokenizer, esm_tokenizer, target_col)
        val_dataset = EnzymeKineticsDataset(df_val, smiles_tokenizer, esm_tokenizer, target_col,
                                            ph_scaler=train_dataset.scaler)
        test_dataset = EnzymeKineticsDataset(df_test, smiles_tokenizer, esm_tokenizer, target_col,
                                             ph_scaler=train_dataset.scaler)

        scaler_path = os.path.join(save_dir, f"fold_{fold}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(train_dataset.scaler, f)

        for split_name, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
            feats = extract_features_for_dataset(
                smiles_encoder, prot_encoder, dataset, batch_size=batch_size,
                device=device, desc=f"Fold{fold} {split_name}"
            )
            np.save(os.path.join(save_dir, f"fold_{fold}_{split_name}_feat.npy"), feats['feat'])
            np.save(os.path.join(save_dir, f"fold_{fold}_{split_name}_label.npy"), feats['label'])
            print(f"  {split_name}: feat{feats['feat'].shape}, label{feats['label'].shape}")
            assert feats['feat'].shape[1] == TOTAL_DIM

    print(f"\nAll features saved to {save_dir}")
