import pandas as pd

# Load dataset
#df = pd.read_csv("/mnt/data/S08.071.dataset.csv")
df = pd.read_csv("./S08.071.dataset.csv")

# Check class distribution
class_counts = df["label"].value_counts().sort_index()

# Calculate ratio
total = len(df)
ratios = class_counts / total

# If imbalance > threshold (e.g., 60/40), apply undersampling
imbalance = max(ratios) > 0.6

balanced_path = None

if imbalance:
    # Separate classes
    df_pos = df[df["label"] == 1]
    df_neg = df[df["label"] == 0]
    
    # Undersample majority
    min_size = min(len(df_pos), len(df_neg))
    
    df_pos_sampled = df_pos.sample(min_size, random_state=42)
    df_neg_sampled = df_neg.sample(min_size, random_state=42)
    
    df_balanced = pd.concat([df_pos_sampled, df_neg_sampled]).sample(frac=1, random_state=42)
    
    #balanced_path = "/mnt/data/S08.071.dataset.balanced.csv"
    balanced_path = "./S08.071.dataset.balanced.csv"
    df_balanced.to_csv(balanced_path, index=False)

class_counts, ratios, imbalance, balanced_path

