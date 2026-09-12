

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Load dataset
#df = pd.read_csv("/mnt/data/S08.071.dataset.csv")
#df = pd.read_csv("./S08.071.dataset.csv")
df = pd.read_csv("./S01.247.dataset.csv")


# Stratified K-Fold
k = 3#5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

paths = []

for i, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    train_fold = df.iloc[train_idx]
    val_fold = df.iloc[val_idx]
    
    #train_path = f"/mnt/data/S08.071.fold{i+1}.train.csv"
    #val_path = f"/mnt/data/S08.071.fold{i+1}.val.csv"
    #train_path = f"./S08.071.fold{i+1}.train.csv"
    #val_path = f"./S08.071.fold{i+1}.val.csv"
    train_path = f"./S01.247.fold{i+1}.train.csv"
    val_path = f"./S01.247.fold{i+1}.val.csv"


    train_fold.to_csv(train_path, index=False)
    val_fold.to_csv(val_path, index=False)
    
    paths.append((train_path, val_path))

paths

