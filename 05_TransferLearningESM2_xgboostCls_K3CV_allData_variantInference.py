#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frozen ESM2 + XGBoost Classifier for S08.071 K=3 evaluation.

Directory expected beside this script:
  ./dataset_k3_backup20260629v03_OK/
    S08.071.fold1ofK3.train.csv
    S08.071.fold1ofK3.val.csv
    S08.071.fold2ofK3.train.csv
    S08.071.fold2ofK3.val.csv
    S08.071.fold3ofK3.train.csv
    S08.071.fold3ofK3.val.csv
    S08.071.train.csv
    S08.071.test.csv
    S08.071.dataset.csv

Each CSV must contain:
  seq,label

Evaluation flow:
  1) Train fold1.train -> validate fold1.val
  2) Train fold2.train -> validate fold2.val
  3) Train fold3.train -> validate fold3.val
  4) Train final model with S08.071.train.csv
  5) Evaluate final model with S08.071.test.csv
  6) After evaluation, train an all-data model with S08.071.dataset.csv
  7) Use the all-data model for variant Spike sliding-window inference

Important:
  ESM2 is NOT fine-tuned. ESM2 is used only as a frozen feature extractor.
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# CONFIG
# =========================================================

PROTEASE = "S08.071"
PROTEASE_NUM = 741
K_FOLD = 3

DATASET_DIR = "./dataset_k3_backup20260629v03_OK"
OUTPUT_DIR = f"./outputdir/frozen_esm2_xgboost_k{K_FOLD}/{PROTEASE_NUM}_{PROTEASE}"

# 軽量: facebook/esm2_t6_8M_UR50D
# 標準: facebook/esm2_t12_35M_UR50D
# 強め: facebook/esm2_t30_150M_UR50D
ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

MAX_LEN = 512
BATCH_SIZE = 32
RANDOM_STATE = 42
N_JOBS_XGB = 4
THRESHOLD = 0.5

SAVE_EMBEDDINGS = True
FORCE_REENCODE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
esm_model = None


# =========================================================
# PATH / SAVE UTILITIES
# =========================================================

def resolve_path(path: str, script_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(script_dir, path))


def make_dirs(output_dir: str) -> Dict[str, str]:
    dirs = {
        "root": output_dir,
        "embeddings": os.path.join(output_dir, "embeddings"),
        "fold_models": os.path.join(output_dir, "fold_models"),
        "final_model": os.path.join(output_dir, "final_model"),
        "all_data_model": os.path.join(output_dir, "all_data_model"),
        "metrics": os.path.join(output_dir, "metrics"),
        "predictions": os.path.join(output_dir, "predictions"),
        "variant_predictions": os.path.join(output_dir, "variant_predictions"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print("Saved:", path)


def csv_stem(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".csv"):
        name = name[:-4]
    return name.replace(".", "_")


# =========================================================
# DATA
# =========================================================

def clean_sequence(seq: str) -> str:
    """
    ESM2に入れるための配列整形。
    '-' は160-mer端のpadding由来と考えて除去する。
    想定外文字は X にする。
    """
    seq = str(seq).strip().upper()
    seq = seq.replace(" ", "")
    seq = seq.replace("\n", "")
    seq = seq.replace("\r", "")
    seq = seq.replace("\t", "")
    seq = seq.replace("-", "")

    allowed = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    seq = "".join([aa if aa in allowed else "X" for aa in seq])
    return seq


def load_seq_label_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "seq" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain 'seq' and 'label' columns.")

    df = df[["seq", "label"]].dropna().reset_index(drop=True)
    df["seq"] = df["seq"].map(clean_sequence)
    df = df[df["seq"].str.len() > 0].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    labels = set(df["label"].unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"label must be 0 or 1. Found: {labels}")

    return df


def print_dataset_info(name: str, df: pd.DataFrame) -> None:
    print(f"[{name}] n = {len(df)}")
    print(f"[{name}] label counts:")
    print(df["label"].value_counts().sort_index())


# =========================================================
# FROZEN ESM2 ENCODING
# =========================================================

def load_frozen_esm2(model_name: str) -> None:
    global tokenizer, esm_model

    print("DEVICE:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = AutoModel.from_pretrained(model_name).to(DEVICE)

    # ここが重要: ESM2はfine-tuningしない。
    for p in esm_model.parameters():
        p.requires_grad = False

    esm_model.eval()
    print("Loaded frozen ESM2:", model_name)
    print("ESM2 parameters frozen: True")


def mean_pool_esm_embeddings(outputs, input_ids, attention_mask):
    """last_hidden_stateから pad/cls/eos を除いてmean poolingする。"""
    token_embeddings = outputs.last_hidden_state  # [B, L, H]
    mask = attention_mask.clone().bool()

    special_token_ids = []
    if tokenizer.pad_token_id is not None:
        special_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.cls_token_id is not None:
        special_token_ids.append(tokenizer.cls_token_id)
    if tokenizer.eos_token_id is not None:
        special_token_ids.append(tokenizer.eos_token_id)

    for token_id in special_token_ids:
        mask = mask & (input_ids != token_id)

    mask = mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1.0)
    return summed / lengths


def encode_sequences_with_esm2(sequences: List[str], batch_size: int, max_len: int) -> np.ndarray:
    if tokenizer is None or esm_model is None:
        raise RuntimeError("Call load_frozen_esm2() before encoding.")

    sequences = [clean_sequence(s) for s in sequences]
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="ESM2 encoding"):
        batch = sequences[i : i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )

        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        # ここも重要: 勾配計算なし。
        with torch.no_grad():
            outputs = esm_model(input_ids=input_ids, attention_mask=attention_mask)
            emb = mean_pool_esm_embeddings(outputs, input_ids, attention_mask)

        all_embeddings.append(emb.detach().cpu().numpy())

    return np.vstack(all_embeddings)


def encode_df_with_cache(
    df: pd.DataFrame,
    csv_path: str,
    embedding_dir: str,
    model_name: str,
    batch_size: int,
    max_len: int,
    force_reencode: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """CSVごとにESM2埋め込みをnpy保存して、再実行時に使い回す。"""
    y = df["label"].to_numpy(dtype=int)
    stem = csv_stem(csv_path)
    model_tag = model_name.replace("/", "_")

    emb_path = os.path.join(embedding_dir, f"{stem}_{model_tag}_embeddings.npy")
    label_path = os.path.join(embedding_dir, f"{stem}_labels.npy")
    clean_csv_path = os.path.join(embedding_dir, f"{stem}_cleaned.csv")

    if (
        SAVE_EMBEDDINGS
        and not force_reencode
        and os.path.exists(emb_path)
        and os.path.exists(label_path)
    ):
        print("Load cached embeddings:", emb_path)
        X = np.load(emb_path)
        y_cached = np.load(label_path)
        if len(y_cached) == len(y):
            return X, y_cached
        print("Cache size mismatch. Re-encode.")

    X = encode_sequences_with_esm2(df["seq"].tolist(), batch_size=batch_size, max_len=max_len)

    if SAVE_EMBEDDINGS:
        np.save(emb_path, X)
        np.save(label_path, y)
        df.to_csv(clean_csv_path, index=False)
        print("Saved embeddings:", emb_path)

    return X, y


# =========================================================
# XGBOOST / METRICS
# =========================================================

def scale_pos_weight(y: np.ndarray) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return float(neg / max(pos, 1))


def build_xgb_classifier(y_train: np.ndarray, random_state: int, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=scale_pos_weight(y_train),
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[dict, np.ndarray]:
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    if len(np.unique(y_true)) >= 2:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    metrics = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, y_pred


def save_prediction_csv(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    dataset_name: str,
    fold: int = None,
) -> None:
    out = pd.DataFrame({
        "dataset": dataset_name,
        "row_index": np.arange(len(df)),
        "seq": df["seq"].values,
        "label": df["label"].values.astype(int),
        "pred_prob_label1": y_prob,
        "pred_label": y_pred.astype(int),
    })
    if fold is not None:
        out.insert(1, "fold", fold)

    out.to_csv(save_path, index=False)
    print("Saved:", save_path)


def save_report_txt(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved:", save_path)


# =========================================================
# TRAINING / EVALUATION
# =========================================================

def train_and_eval_one_fold(
    fold: int,
    train_csv: str,
    val_csv: str,
    dirs: Dict[str, str],
    args,
) -> dict:
    print("=" * 80)
    print(f"Fold {fold}/{args.k_fold}")
    print("=" * 80)

    train_df = load_seq_label_csv(train_csv)
    val_df = load_seq_label_csv(val_csv)

    print_dataset_info(f"fold{fold}_train", train_df)
    print_dataset_info(f"fold{fold}_val", val_df)

    X_train, y_train = encode_df_with_cache(
        train_df,
        train_csv,
        dirs["embeddings"],
        args.esm_model_name,
        args.batch_size,
        args.max_len,
        args.force_reencode,
    )
    X_val, y_val = encode_df_with_cache(
        val_df,
        val_csv,
        dirs["embeddings"],
        args.esm_model_name,
        args.batch_size,
        args.max_len,
        args.force_reencode,
    )

    clf = build_xgb_classifier(y_train, random_state=args.random_state + fold, n_jobs=args.n_jobs_xgb)
    clf.fit(X_train, y_train, verbose=False)

    y_prob = clf.predict_proba(X_val)[:, 1]
    metrics, y_pred = calc_metrics(y_val, y_prob, args.threshold)

    metrics.update({
        "fold": int(fold),
        "train_csv": train_csv,
        "val_csv": val_csv,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "train_neg": int((y_train == 0).sum()),
        "train_pos": int((y_train == 1).sum()),
        "val_neg": int((y_val == 0).sum()),
        "val_pos": int((y_val == 1).sum()),
        "scale_pos_weight": float(scale_pos_weight(y_train)),
    })

    print(json.dumps(metrics, indent=2))

    model_path = os.path.join(dirs["fold_models"], f"xgboost_frozenESM2_fold{fold}.joblib")
    joblib.dump(clf, model_path)
    print("Saved:", model_path)

    save_json(metrics, os.path.join(dirs["metrics"], f"fold{fold}_validation_metrics.json"))
    save_prediction_csv(
        val_df,
        y_prob,
        y_pred,
        os.path.join(dirs["predictions"], f"fold{fold}_validation_predictions.csv"),
        dataset_name=f"fold{fold}_validation",
        fold=fold,
    )
    save_report_txt(
        y_val,
        y_pred,
        os.path.join(dirs["metrics"], f"fold{fold}_classification_report.txt"),
    )

    return metrics


def summarize_cv(fold_metrics: List[dict], dirs: Dict[str, str]) -> None:
    df = pd.DataFrame(fold_metrics)
    fold_csv = os.path.join(dirs["metrics"], "cv_fold_metrics.csv")
    df.to_csv(fold_csv, index=False)
    print("Saved:", fold_csv)

    metric_cols = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
    ]

    rows = []
    for col in metric_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "metric": col,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })

    summary = pd.DataFrame(rows)
    summary_csv = os.path.join(dirs["metrics"], "cv_metrics_summary_mean_std.csv")
    summary.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)
    print(summary.to_string(index=False))


def train_final_and_test(final_train_csv: str, test_csv: str, dirs: Dict[str, str], args) -> dict:
    print("=" * 80)
    print("Final model: S08.071.train.csv -> S08.071.test.csv")
    print("=" * 80)

    train_df = load_seq_label_csv(final_train_csv)
    test_df = load_seq_label_csv(test_csv)

    print_dataset_info("final_train", train_df)
    print_dataset_info("test", test_df)

    X_train, y_train = encode_df_with_cache(
        train_df,
        final_train_csv,
        dirs["embeddings"],
        args.esm_model_name,
        args.batch_size,
        args.max_len,
        args.force_reencode,
    )
    X_test, y_test = encode_df_with_cache(
        test_df,
        test_csv,
        dirs["embeddings"],
        args.esm_model_name,
        args.batch_size,
        args.max_len,
        args.force_reencode,
    )

    clf = build_xgb_classifier(y_train, random_state=args.random_state, n_jobs=args.n_jobs_xgb)
    clf.fit(X_train, y_train, verbose=False)

    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics, y_pred = calc_metrics(y_test, y_prob, args.threshold)

    metrics.update({
        "dataset": "test",
        "final_train_csv": final_train_csv,
        "test_csv": test_csv,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_neg": int((y_train == 0).sum()),
        "train_pos": int((y_train == 1).sum()),
        "test_neg": int((y_test == 0).sum()),
        "test_pos": int((y_test == 1).sum()),
        "scale_pos_weight": float(scale_pos_weight(y_train)),
    })

    print(json.dumps(metrics, indent=2))

    final_model_path = os.path.join(dirs["final_model"], "xgboost_frozenESM2_final_S08_071.joblib")
    joblib.dump(clf, final_model_path)
    print("Saved:", final_model_path)

    save_json(metrics, os.path.join(dirs["metrics"], "test_metrics.json"))
    save_prediction_csv(
        test_df,
        y_prob,
        y_pred,
        os.path.join(dirs["predictions"], "test_predictions.csv"),
        dataset_name="test",
        fold=None,
    )
    save_report_txt(
        y_test,
        y_pred,
        os.path.join(dirs["metrics"], "test_classification_report.txt"),
    )

    return metrics


# ========================================================
#    ORIGINAL
# =========================================================

origin_fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
def test_data_gen_origin():
	return origin_fullseq
# =========================================================
# VARIANTS
# =========================================================

def test_data_gen_alpha():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAISGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIDDTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPINFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILARLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTHNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_beta():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFANPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRGLPQGFSALEPLVDLPIGINITRFQTLHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGVEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGVENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_gamma():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNFTNRTQLPSAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNYPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGTIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEYVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAAIKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASFVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_delta():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLRTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASIEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLDVYYHKNNKSWMESGVYSSANNCTFEYVSQPFLMDLEVKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSRRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQNVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFLTQRNFYEPQTITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_omicron():
    fullseq_mutation = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVFRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGVNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEYVNNSYECDIPIGAGICASYQTQTKSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNHNAQALNTLVKQLSSKFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    return fullseq_mutation

def test_data_gen_omicron_ba1():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHVISGTNGTKRFDNPVLPFNDGVYFASIEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLDHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPIIVREPEDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLKGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEYVNNSYECDIPIGAGICASYQTQTKSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLKRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKYFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFKGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNHNAQALNTLVKQLSSKFGAISSVLNDIFSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation

def test_data_gen_omicron_ba2():
    fullseq_mutation = 'MFVFLVLLPLVSSQCVNLITRTQSYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLDVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLGRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNXAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEYVNNSYECDIPIGAGICASYQTQTKSHRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLKRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKYFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNHNAQALNTLVKQLSSKFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    return fullseq_mutation



# =========================================================
# VARIANT TABLE
# =========================================================

variant_functions = [
    ("origin", test_data_gen_origin),
    ("alpha", test_data_gen_alpha),
    ("beta", test_data_gen_beta),
    ("gamma", test_data_gen_gamma),
    ("delta", test_data_gen_delta),
    ("omicron", test_data_gen_omicron),
    ("omicron_ba1", test_data_gen_omicron_ba1),
    ("omicron_ba2", test_data_gen_omicron_ba2),
]


# =========================================================
# ALL-DATA FINAL MODEL AND VARIANT WINDOW INFERENCE
# =========================================================

def furin_motif_score(seq: str) -> Tuple[float, str, int]:
    """
    Furin-like motif R-X-[K/R]-R を簡易スコア化する。
    XGBoostの予測とは別の解釈用列として使う。
    """
    seq = clean_sequence(seq)
    best_score = 0.0
    best_motif = ""
    best_pos = -1
    pattern = re.compile(r"R.[KR]R")

    for m in pattern.finditer(seq):
        motif = m.group()
        start = m.start()
        score = 0.80

        left = max(0, start - 2)
        right = min(len(seq), start + 6)
        local = seq[left:right]
        rk_count = local.count("R") + local.count("K")
        score += min(0.20, 0.04 * rk_count)

        if motif == "RRAR":
            score = 1.0

        if score > best_score:
            best_score = score
            best_motif = motif
            best_pos = start

    return float(best_score), best_motif, int(best_pos)


def make_sliding_windows(fullseq: str, trim_num: int) -> List[dict]:
    """fullseqからtrim_num aaのsliding windowを1 aaずつ作る。"""
    fullseq = clean_sequence(fullseq)
    if len(fullseq) < trim_num:
        raise ValueError(f"Sequence length {len(fullseq)} is shorter than n_mer={trim_num}.")

    windows = []
    for loc in range(0, len(fullseq) - trim_num + 1):
        windows.append({
            "window_index": loc,
            "start_0based": loc,
            "end_0based_exclusive": loc + trim_num,
            "start_1based": loc + 1,
            "end_1based": loc + trim_num,
            "center_1based": loc + trim_num // 2 + 1,
            "sequence": fullseq[loc : loc + trim_num],
        })
    return windows


def extract_reference_window(fullseq: str, motif: str, trim_num: int) -> dict:
    """origin配列からS1/S2 reference motif周辺のwindowを作る。"""
    fullseq = clean_sequence(fullseq)
    motif_start = fullseq.find(motif)
    if motif_start < 0:
        raise ValueError(f"Reference motif not found: {motif}")

    cleavage_like_pos = motif_start + motif.find("SV")
    start = max(0, cleavage_like_pos - trim_num // 2)
    end = start + trim_num

    if end > len(fullseq):
        end = len(fullseq)
        start = max(0, end - trim_num)

    ref_seq = fullseq[start:end]
    if len(ref_seq) != trim_num:
        raise ValueError(f"Reference window length is not {trim_num}: {len(ref_seq)}")

    return {
        "sequence": ref_seq,
        "start_1based": start + 1,
        "end_1based": end,
        "center_1based": start + trim_num // 2 + 1,
        "motif_start_1based": motif_start + 1,
    }


def cosine_similarity_to_reference(embeddings: np.ndarray, reference_embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """各window embeddingとreference embeddingのcosine similarityを計算する。"""
    embeddings_norm = embeddings / np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-8,
        None,
    )
    ref = reference_embedding.reshape(1, -1)
    ref_norm = ref / np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-8, None)
    cosine = np.sum(embeddings_norm * ref_norm, axis=1)
    score_01 = (cosine + 1.0) / 2.0
    return cosine, score_01


def train_all_dataset_model(all_dataset_csv: str, dirs: Dict[str, str], args) -> XGBClassifier:
    """
    S08.071.dataset.csv、つまりtrain+testを全部合わせた全データで
    最終的な変異株推論用ESM2+XGBoost Classifierを学習する。

    注意:
      このモデルは、テスト評価後の実運用・推論用モデル。
      汎化性能の報告には、前段のS08.071.train.csv -> S08.071.test.csvの結果を使う。
    """
    print("=" * 80)
    print("All-data model: S08.071.dataset.csv -> variant window inference model")
    print("=" * 80)

    df = load_seq_label_csv(all_dataset_csv)
    print_dataset_info("all_dataset", df)

    X_all, y_all = encode_df_with_cache(
        df,
        all_dataset_csv,
        dirs["embeddings"],
        args.esm_model_name,
        args.batch_size,
        args.max_len,
        args.force_reencode,
    )

    clf = build_xgb_classifier(y_all, random_state=args.random_state, n_jobs=args.n_jobs_xgb)
    clf.fit(X_all, y_all, verbose=False)

    model_path = os.path.join(dirs["all_data_model"], "xgboost_frozenESM2_all_dataset_S08_071.joblib")
    joblib.dump(clf, model_path)
    print("Saved:", model_path)

    config = {
        "all_dataset_csv": all_dataset_csv,
        "n": int(len(y_all)),
        "neg": int((y_all == 0).sum()),
        "pos": int((y_all == 1).sum()),
        "scale_pos_weight": float(scale_pos_weight(y_all)),
        "model_path": model_path,
        "note": "This all-data model is for final variant-window inference after model evaluation is finished.",
    }
    save_json(config, os.path.join(dirs["all_data_model"], "all_dataset_model_config.json"))
    return clf


def predict_one_variant_windows(
    mutation_name: str,
    fullseq: str,
    model: XGBClassifier,
    reference_embedding: np.ndarray,
    dirs: Dict[str, str],
    args,
) -> dict:
    windows = make_sliding_windows(fullseq, args.n_mer)
    sequences = [w["sequence"] for w in windows]

    print("=" * 80)
    print(f"Variant inference: {mutation_name}")
    print("=" * 80)
    print("Total windows:", len(sequences))

    embeddings = encode_sequences_with_esm2(sequences, batch_size=args.batch_size, max_len=args.max_len)
    prob = model.predict_proba(embeddings)[:, 1]
    pred = (prob >= args.threshold).astype(int)
    cosine, esm_score = cosine_similarity_to_reference(embeddings, reference_embedding)

    rows = []
    for idx, w in enumerate(windows):
        motif_score, motif, motif_pos_0based = furin_motif_score(w["sequence"])
        motif_pos_1based = -1
        if motif_pos_0based >= 0:
            motif_pos_1based = w["start_1based"] + motif_pos_0based

        rows.append({
            "variant": mutation_name,
            "window_index": int(w["window_index"]),
            "start_1based": int(w["start_1based"]),
            "end_1based": int(w["end_1based"]),
            "center_1based": int(w["center_1based"]),
            "sequence": w["sequence"],
            "xgb_pred_prob_label1": float(prob[idx]),
            "xgb_pred_label": int(pred[idx]),
            "cosine_to_origin_s1s2_reference": float(cosine[idx]),
            "esm_similarity_score_01": float(esm_score[idx]),
            "furin_motif_score": float(motif_score),
            "best_furin_motif": motif,
            "best_furin_motif_pos_0based_in_window": int(motif_pos_0based),
            "best_furin_motif_pos_1based_in_spike": int(motif_pos_1based),
        })

    result_df = pd.DataFrame(rows)
    pred_path = os.path.join(
        dirs["variant_predictions"],
        f"predict_sprotein_windows_allDataModel_{mutation_name}.csv",
    )
    result_df.to_csv(pred_path, index=False)
    print("Saved:", pred_path)

    if SAVE_EMBEDDINGS:
        emb_path = os.path.join(
            dirs["variant_predictions"],
            f"embeddings_variant_windows_{mutation_name}.npy",
        )
        np.save(emb_path, embeddings)
        print("Saved:", emb_path)

    best_idx = int(np.argmax(prob))
    best = rows[best_idx]
    summary = {
        "variant": mutation_name,
        "num_windows": int(len(rows)),
        "num_pred_positive_windows": int(pred.sum()),
        "max_xgb_pred_prob_label1": float(best["xgb_pred_prob_label1"]),
        "max_xgb_pred_label": int(best["xgb_pred_label"]),
        "max_window_index": int(best["window_index"]),
        "max_window_start_1based": int(best["start_1based"]),
        "max_window_end_1based": int(best["end_1based"]),
        "max_window_center_1based": int(best["center_1based"]),
        "max_esm_similarity_score_01": float(best["esm_similarity_score_01"]),
        "max_furin_motif_score": float(best["furin_motif_score"]),
        "best_furin_motif": best["best_furin_motif"],
        "best_furin_motif_pos_1based_in_spike": int(best["best_furin_motif_pos_1based_in_spike"]),
        "max_window_sequence": best["sequence"],
        "prediction_csv": pred_path,
    }
    print("Best window summary:")
    print(json.dumps(summary, indent=2))
    return summary


def run_variant_window_inference(all_data_model: XGBClassifier, dirs: Dict[str, str], args) -> List[dict]:
    """全データ学習モデルで、登録済み変異株Spike配列を160-mer window推論する。"""
    reference = extract_reference_window(
        fullseq=origin_fullseq,
        motif=args.reference_motif,
        trim_num=args.n_mer,
    )
    print("Reference S1/S2 window:")
    print(json.dumps(reference, indent=2))

    reference_embedding = encode_sequences_with_esm2(
        [reference["sequence"]],
        batch_size=args.batch_size,
        max_len=args.max_len,
    )[0]

    summary_rows = []
    for mutation_name, seq_func in variant_functions:
        fullseq = seq_func()
        summary = predict_one_variant_windows(
            mutation_name=mutation_name,
            fullseq=fullseq,
            model=all_data_model,
            reference_embedding=reference_embedding,
            dirs=dirs,
            args=args,
        )
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(dirs["variant_predictions"], "summary_variant_window_predictions_allDataModel.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
    return summary_rows


# =========================================================
# MAIN
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--protease", type=str, default=PROTEASE)
    parser.add_argument("--k_fold", type=int, default=K_FOLD)
    parser.add_argument("--esm_model_name", type=str, default=ESM_MODEL_NAME)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE)
    parser.add_argument("--n_jobs_xgb", type=int, default=N_JOBS_XGB)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--n_mer", type=int, default=160)
    parser.add_argument("--reference_motif", type=str, default="PRRARSVAS")
    parser.add_argument("--force_reencode", action="store_true")
    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.dataset_dir = resolve_path(args.dataset_dir, script_dir)
    args.output_dir = resolve_path(args.output_dir, script_dir)
    args.force_reencode = bool(args.force_reencode or FORCE_REENCODE)

    dirs = make_dirs(args.output_dir)

    print("Dataset dir:", args.dataset_dir)
    print("Output dir:", args.output_dir)
    print("Protease:", args.protease)
    print("K fold:", args.k_fold)

    # file paths
    fold_train_csvs = []
    fold_val_csvs = []
    for fold in range(1, args.k_fold + 1):
        fold_train_csvs.append(
            os.path.join(args.dataset_dir, f"{args.protease}.fold{fold}ofK{args.k_fold}.train.csv")
        )
        fold_val_csvs.append(
            os.path.join(args.dataset_dir, f"{args.protease}.fold{fold}ofK{args.k_fold}.val.csv")
        )

    final_train_csv = os.path.join(args.dataset_dir, f"{args.protease}.train.csv")
    test_csv = os.path.join(args.dataset_dir, f"{args.protease}.test.csv")
    all_dataset_csv = os.path.join(args.dataset_dir, f"{args.protease}.dataset.csv")

    required_files = fold_train_csvs + fold_val_csvs + [final_train_csv, test_csv, all_dataset_csv]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    run_config = {
        "protease": args.protease,
        "k_fold": args.k_fold,
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "esm_model_name": args.esm_model_name,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "random_state": args.random_state,
        "threshold": args.threshold,
        "device": DEVICE,
        "esm2_frozen": True,
        "fold_train_csvs": fold_train_csvs,
        "fold_val_csvs": fold_val_csvs,
        "final_train_csv": final_train_csv,
        "test_csv": test_csv,
        "all_dataset_csv": all_dataset_csv,
        "n_mer": args.n_mer,
        "reference_motif": args.reference_motif,
    }
    save_json(run_config, os.path.join(dirs["root"], "run_config.json"))

    # ESM2 frozen encoder
    load_frozen_esm2(args.esm_model_name)

    # K=3 validation
    fold_metrics = []
    for fold in range(1, args.k_fold + 1):
        m = train_and_eval_one_fold(
            fold=fold,
            train_csv=fold_train_csvs[fold - 1],
            val_csv=fold_val_csvs[fold - 1],
            dirs=dirs,
            args=args,
        )
        fold_metrics.append(m)

    summarize_cv(fold_metrics, dirs)

    # Final train and test
    test_metrics = train_final_and_test(final_train_csv, test_csv, dirs, args)

    # After all evaluations are finished, train with all data and infer variant windows.
    all_data_model = train_all_dataset_model(all_dataset_csv, dirs, args)
    variant_summary_rows = run_variant_window_inference(all_data_model, dirs, args)

    final_summary = {
        "cv_fold_metrics_csv": os.path.join(dirs["metrics"], "cv_fold_metrics.csv"),
        "cv_metrics_summary_csv": os.path.join(dirs["metrics"], "cv_metrics_summary_mean_std.csv"),
        "test_metrics_json": os.path.join(dirs["metrics"], "test_metrics.json"),
        "test_predictions_csv": os.path.join(dirs["predictions"], "test_predictions.csv"),
        "final_model": os.path.join(dirs["final_model"], "xgboost_frozenESM2_final_S08_071.joblib"),
        "all_data_model": os.path.join(dirs["all_data_model"], "xgboost_frozenESM2_all_dataset_S08_071.joblib"),
        "variant_summary_csv": os.path.join(dirs["variant_predictions"], "summary_variant_window_predictions_allDataModel.csv"),
        "test_metrics": test_metrics,
        "variant_summary_rows": variant_summary_rows,
    }
    save_json(final_summary, os.path.join(dirs["root"], "final_summary.json"))

    elapsed = time.time() - start
    print()
    print("Finished.")
    print(f"Time (sec): {elapsed:.2f}")
    print(f"Time (min): {elapsed / 60:.2f}")
    print("END.")


if __name__ == "__main__":
    main()
