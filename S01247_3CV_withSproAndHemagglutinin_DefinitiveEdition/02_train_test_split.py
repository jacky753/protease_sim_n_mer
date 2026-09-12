

import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
#dataset_path = "/mnt/data/S08.071.dataset.csv"
#dataset_path = "./S08.071.dataset.csv"
dataset_path = "./S01.247.dataset.csv"
df = pd.read_csv(dataset_path)

# Train/test split (80/20) with stratification
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Save files
#train_path = "/mnt/data/S08.071.train.csv"
#test_path = "/mnt/data/S08.071.test.csv"
#train_path = "./S08.071.train.csv"
#test_path = "./S08.071.test.csv"
train_path = "./S01.247.train.csv"
test_path = "./S01.247.test.csv"


train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

train_path, test_path


