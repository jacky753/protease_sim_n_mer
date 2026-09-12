import csv


inout_file_dic = {
    "S01.247.dataset.csv": "S01.247.dataset.fasta",
    "S01.247.test.csv": "S01.247.test.fasta",
    "S01.247.train.csv": "S01.247.train.fasta",
    "S01.247.fold1.train.csv": "S01.247.fold1.train.fasta",
    "S01.247.fold1.val.csv": "S01.247.fold1.val.fasta",
    "S01.247.fold2.train.csv": "S01.247.fold2.train.fasta",
    "S01.247.fold2.val.csv": "S01.247.fold2.val.fasta",
    "S01.247.fold3.train.csv": "S01.247.fold3.train.fasta",
    "S01.247.fold3.val.csv": "S01.247.fold3.val.fasta"
    }

#input_file = "S01.247.dataset.csv"
#output_file = "S01.247.dataset.fasta"

#input_file = "test-data_S01.247_seqs.csv"
#output_file = "test-data_S01.247_seqs.fasta"
#input_file = "learn-data_S01.247_seqs.csv"
#output_file = "learn-data_S01.247_seqs.fasta"

#input_file = "train-dataS01.247_cv0.csv"
#output_file = "train-dataS01.247_cv0.fasta"
#input_file = "val-dataS01.247_cv0.csv"
#output_file = "val-dataS01.247_cv0.fasta"

#input_file = "train-dataS01.247_cv1.csv"
#output_file = "train-dataS01.247_cv1.fasta"
#input_file = "val-dataS01.247_cv1.csv"
#output_file = "val-dataS01.247_cv1.fasta"

#input_file = "train-dataS01.247_cv2.csv"
#output_file = "train-dataS01.247_cv2.fasta"
#input_file = "val-dataS01.247_cv2.csv"
#output_file = "val-dataS01.247_cv2.fasta"

for  i in range(len(inout_file_dic)):
    input_file = list(inout_file_dic.keys())[i]
    output_file = inout_file_dic[input_file]
    with open(input_file, "r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)

        with open(output_file, "w", encoding="utf-8") as outfile:
            for i, row in enumerate(reader, start=1):
                sequence = row["seq"].strip()
                label = row["label"].strip()

                outfile.write(f">seq_{i} label={label}\n")
                outfile.write(f"{sequence}\n")

    print(f"FASTA file was created: {output_file}")
print("END.")
