import pandas as pd

# File paths
#pos_path = "/mnt/data/cleave_pattern_one_letter_aa_S08.071.csv"
#neg_path = "/mnt/data/negative_pattern_one_letter_aa_S08.071.csv"
pos_path = "./cleave_pattern_one_letter_aa_S08.071.csv"
neg_path = "./negative_pattern_one_letter_aa_S08.071.csv"
#pos_path = "./cleave_pattern_one_letter_aa_S01.247.csv"
#neg_path = "./negative_pattern_one_letter_aa_S01.247.csv"


# Load data
pos_df = pd.read_csv(pos_path)
neg_df = pd.read_csv(neg_path)

# IDs to remove
#remove_ids = {"P11223", "P0DTC2", "P59594", "K9N5Q8"}
remove_ids = {"XXXXXX"}

# Extract and remove rows with specified IDs (positive)
pos_removed = pos_df[pos_df["uniprot_id"].isin(remove_ids)]
pos_clean = pos_df[~pos_df["uniprot_id"].isin(remove_ids)]

# Extract and remove rows with specified IDs (negative)
neg_removed = neg_df[neg_df["uniprot_id"].isin(remove_ids)]
neg_clean = neg_df[~neg_df["uniprot_id"].isin(remove_ids)]

# Save removed rows
#pos_removed_path = "/mnt/data/removed_positive.csv"
#neg_removed_path = "/mnt/data/removed_negative.csv"
pos_removed_path = "./removed_positive.csv"
neg_removed_path = "./removed_negative.csv"

pos_removed.to_csv(pos_removed_path, index=False)
neg_removed.to_csv(neg_removed_path, index=False)

# Create dataset
dataset = pd.DataFrame(columns=["seq", "label"])

# Add positive data
pos_sequences = pos_clean["cleave_pattern"].dropna()
pos_data = pd.DataFrame({
    "seq": pos_sequences,
    "label": 1
})

# Add negative data
neg_sequences = neg_clean["negative_pattern"].dropna()
neg_data = pd.DataFrame({
    "seq": neg_sequences,
    "label": 0
})

# Combine
dataset = pd.concat([pos_data, neg_data], ignore_index=True)

# Save dataset
#dataset_path = "/mnt/data/S08.071.dataset.csv"
dataset_path = "./S08.071.dataset.csv"
#dataset_path = "./S01.247.dataset.csv"
dataset.to_csv(dataset_path, index=False)

pos_removed_path, neg_removed_path, dataset_path

