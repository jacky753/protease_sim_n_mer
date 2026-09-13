#protein-bert-seq_RF-classifier_ALL-Human-proteases_into_4-proteases_for_mutations_16-mer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import csv
from pandas import plotting 
import urllib.request 
import matplotlib.ticker as ticker
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
from proteinbert import load_pretrained_model
import tensorflow as tf

from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
#from msilib import Feature
from tkinter import XView
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import urllib.request
import sys
import os, glob
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch.optim as optim
import json
import statistics
import math
import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.nn import LayerNorm
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import datetime
import scipy
from scipy import signal
from urllib.request import urlopen
#from lxml import etree
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import MySQLdb as mydb
import time
import datetime


charge = {"A":0., "C":-0.0735876, "D":-0.9994991, "E":-0.9987427, "F":0., "G":0., "H":0.0593509, "I":0.,
          "K":0.9997489, "L":0., "M":0., "N":0., "P":0., "Q":0., "R":0.9999950, "S":0., "T":0., "V":0., "W":0., "Y":-0.0001995}
hydrophobicity = {"A":0.1630295, "C":-0.2554557, "D":-0.9794608, "E":-0.7458280, "F":0.9152760, "G":-0.2554557, "H":-0.7869063, "I":0.8510911, "K":-0.8510911, "L":1.,
                  "M":0.7227214, "N":-1., "P":-0.0860077, "Q":-0.8305520, "R":-0.7021823, "S":-0.4685494, "T":-0.3838254, "V":0.6816431, "W":0.8305520, "Y":0.5738126}

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'  # X is used as a positional gap/unknown residue for ProteinBERT input
class_list = ["Negac", "Posic", "Noc", "Nonp"]
Negac_list = ["D", "E"]
Posic_list = ["R", "K", "H"]
Noc_list = ["N", "Q", "S", "T", "Y"]
Nonp_list = ["A", "G", "V", "L", "I", "P", "F", "M", "W", "C"]

def main():
    
    encoding_mode = 'pbseq'
    trim_num = 160#50
    cur = con_db()
    device = set_device()

    merops_code_mece = create_merops_code_table()
    print("merops_code_mece.")
    #print(merops_code_mece)
    print(len(merops_code_mece))
    n_mer = 160 #50#16
    
    #for k, protease in enumerate(['S01.047', 'S01.021', 'S01.247', 'S01.034', 'S01.087']): 
    #target_name = 'C01.032'
    #target_name ='S01.247'
    target_name ='S08.071'
    protease_num_dic = {'S01.247': 566, 'S08.071': 741, 'C01.032': 62, 
                    'S01.047': 503, 'S01.021': 498, 
                    'S01.034': 502, 
                    'S01.087': 512, 'S01.292': 578}
    # S01.247: TMPRSS2, epitheliasin
    # S08.071: Furin
    # C01.032: Cathepsin L
    # S01.047: HAT, TMPRSS11D
    # S01.021: DESC
    # S01.034: TMPRSS4
    # S01.087: TMPRSS13, subs=0, so need to add manually
    # S01.292: TMPRSS11A, HATL1, subs=0, so need to add manually     
    # protease_num_dic = {'S01.247': 566, 'S08.071': 741}
    k_cv_num = 3
    
    for i, protease in enumerate([target_name]): 
    #for i, protease in enumerate(['S08.071']): 
    #for k, protease in enumerate(['A01.003']): 
        #k = 2
        
        i = i + protease_num_dic[target_name]
        #i = i + protease_num_dic['S08.071']
        print("i: {}".format(i))
        # l = k
        # 作成したいディレクトリのパス
        dir_path = "./outputdir_rf_cls_priority/{}/{}_{}".format(encoding_mode, i, protease)
        # ディレクトリが存在しなければ作成する
        os.makedirs(dir_path, exist_ok=True)

        print("Directory OK")

        print("="*10+"protease: {}".format(protease)+"_protease_turn: {}".format(i)+"="*10)

        merops_code = [protease]

        cur.execute("SELECT uniprot_acc, p1 FROM cleavage where code=(%s);", merops_code)
        subs = cur.fetchall()

        print("="*10+"encoding data"+"="*10)
        pretrained_model_generator02, input_encoder02 = load_pretrained_model()
        df_positive = pd.read_csv('./proteases/cleave_pattern_one_letter_aa_{}.csv'.format(protease))

        seqs02= df_positive['cleave_pattern']
        print("seqs02.")
        print(seqs02)
        seq_len02= len(seqs02[0])+2
        print("seq_len")
        print(seq_len02)
        seqs02 = seqs02.copy()
        seqs02 = seqs02.str.replace('-', '', regex=False)
        lens = seqs02.str.replace('-', '').str.len()
        print(lens[lens > 160])

        seqs02 = seqs02.str.slice(0, seq_len02-2)
        lengths = seqs02.str.len()
        print(lengths.describe())
        print(lengths[lengths > seq_len02].head())


        from proteinbert.model_generation import tokenize_seq

        lengths = []
        bad = []

        for j, s in enumerate(seqs02):
            try:
                t = tokenize_seq(s)
                lengths.append(len(t))
                if len(t) != len(s):
                    bad.append((j, s, len(s), len(t)))
            except Exception as e:
                bad.append((j, s, str(e)))

        print("token length unique:", set(lengths))
        print("bad examples:", bad[:5])


        
        batch_size = 1#4#1
        #model02 = get_model_with_hidden_layers_as_outputs(pretrained_model_generator02.create_model(seq_len02))
        # model02 = pretrained_model_generator02.create_model(seq_len02)
        # encoded_x_positive02 = input_encoder02.encode_X(seqs02, seq_len02)
        # ① hidden 出力込みモデル
        model02 = get_model_with_hidden_layers_as_outputs(
            pretrained_model_generator02.create_model(seq_len02)
        )

        # encoded_x_positive02 = input_encoder02.encode_X(seqs02, seq_len02)
        encoded_seq_posi, encoded_annt_posi = input_encoder02.encode_X(seqs02, seq_len02)

        # ② batch_size=1 で逐次 predict
        local_reps = []

        for j in range(len(encoded_seq_posi)):
            local, global_ = model02.predict(
                [encoded_seq_posi[j:j+1], encoded_annt_posi[j:j+1]], 
                batch_size=1,
                verbose=0
            )
            
            # local: [1, L, D] → [L, D]
            local_reps.append(local[0].astype(np.float32))

        # ③ CPU 側で list として保持
        local_representations_positive = local_reps


        print("model02.")
        print(model02)
        

    




        # local_representations_positive, global_representations_positive = model02.predict(encoded_x_positive02, batch_size=batch_size)
        #local_representations_positive = model02.predict(encoded_x_positive02, batch_size=batch_size)
        print("local_representaions_positive.")
        #print(local_representations_positive.shape)
        print(len(local_representations_positive))
        #print(np.array(local_representations_positive).shape)



        pretrained_model_generator03, input_encoder03 = load_pretrained_model()
        df_negative =  pd.read_csv('./proteases/negative_pattern_one_letter_aa_{}.csv'.format(protease))

        seqs03= df_negative['negative_pattern']
        lengths = [len(s.replace('-', '')) for s in seqs03]
        print("max(lengths), min(lengths): ")
        print(max(lengths), min(lengths))

        print("seqs03 on negative data.")
        print(seqs03)
        seq_len03 = len(seqs03[0])+2
        print("seq_len03.")
        print(seq_len03)
        seqs03 = seqs03.copy()
        seqs03 = seqs03.str.replace('-', '', regex=False)
        seqs03 = seqs03.str.slice(0, seq_len03-2)
        lengths = seqs03.str.len()
        print(lengths.describe())
        print(lengths[lengths > seq_len03].head())

        # model03 = get_model_with_hidden_layers_as_outputs(pretrained_model_generator03.create_model(seq_len03))
        # model03 = pretrained_model_generator03.create_model(seq_len03)
        #encoded_x_negative03 = input_encoder03.encode_X(seqs03, seq_len03)
        model03 = get_model_with_hidden_layers_as_outputs(
            pretrained_model_generator03.create_model(seq_len03)
        )

        # encoded_x_negative03 = input_encoder03.encode_X(seqs03, seq_len03)
        encoded_seq_negative, encoded_annt_negative = input_encoder03.encode_X(seqs03, seq_len03)

        # ② batch_size=1 で逐次 predict
        local_reps = []

        for j in range(len(encoded_seq_negative)):
            local, global_ = model03.predict(
                [encoded_seq_negative[j:j+1], encoded_annt_negative[j:j+1]],
                batch_size=1,
                verbose=0
            )
            
            # local: [1, L, D] → [L, D]
            local_reps.append(local[0].astype(np.float32))

        # ③ CPU 側で list として保持
        local_representations_negative = local_reps

        print("seqs03 in negative.")
        print(len(seqs03))
        print("seq_len03: {}".format(seq_len03))
        print("Try model.predict() in negative data.")
        # local_representations_negative, global_representations_negative = model03.predict(encoded_x_negative03, batch_size=batch_size)
        # local_representations_negative = model03.predict(encoded_x_negative03, batch_size=batch_size)
        print("local_representations_negative.")
        #print(local_representations_negative.shape)
        print(len(local_representations_negative))
        #print(np.array(local_representations_negative).shape)
        



        # create positiev dataset
        cleave_pattern01 = df_positive['cleave_pattern'][0]#df_01.iloc[1, 3]
        uniprot_id01 = df_positive['uniprot_id'][0]
                
        # create positivate data
        print('cleave_pattern01.')
        print(cleave_pattern01)
        df_test = char2vec(cleave_pattern01, n_mer)

        test_array = df_test.to_numpy().flatten()

        df_concat = pd.DataFrame([test_array], index=['{}'.format(uniprot_id01+'+')])
        df_concat['label'] = 1

        test_array_pb_seq = local_representations_positive[0].flatten()
        df_concat_pb_seq = pd.DataFrame([test_array_pb_seq], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_seq['label'] = 1

        # test_array_pb_go = global_representations_positive[0].flatten()
        #df_concat_pb_go = pd.DataFrame([test_array_pb_go], index=['{}'.format(uniprot_id01+'+')])
        #df_concat_pb_go['label'] = 1

        
        
        for j in range(len(df_positive)):
        #for j in range(10):
            print("="*10 + "positive encoding {}/{}".format(j+1, len(df_positive)) + "="*10)
            #print("i: {}".format(i))
            # uniprot_id = df_positive.iloc[i, 1]
            uniprot_id = df_positive['uniprot_id'][j]
            #print("uniprot_id: {}".format(uniprot_id))
            # cleave_pattern = df_positive.iloc[i, 3]
            cleave_pattern = df_positive['cleave_pattern'][j]
            print("len of positive_pattern seq:{}".format(len(cleave_pattern)))

            one_hot_seq = char2vec(cleave_pattern, n_mer)
            one_hot_array = one_hot_seq.to_numpy().flatten()
            pb_seq_array =  local_representations_positive[j].flatten()
            print("pb_seq_array in positive : {}".format(pb_seq_array.shape))
            # pb_go_array =  global_representations_positive[j].flatten()

            df = pd.DataFrame([one_hot_array], index=['{}'.format(uniprot_id)])
            df['label'] = 1
            df_pb_seq = pd.DataFrame([pb_seq_array], index=['{}'.format(uniprot_id)])
            df_pb_seq['label'] = 1
            # df_pb_go = pd.DataFrame([pb_go_array], index=['{}'.format(uniprot_id)])
            # df_pb_go['label'] = 1
            #print("df for df_concat.")
            #print(df)
            #df = pd.DataFrame([one_hot_seq], index=['{}'.format(uniprot_id)])
            #df['label'] = 1
            df_concat = pd.concat([df_concat, df])
            df_concat_pb_seq = pd.concat([df_concat_pb_seq, df_pb_seq])
            # df_concat_pb_go = pd.concat([df_concat_pb_go, df_pb_go])

        print("df_concat.")
        #print(df_concat.drop(df.loc[0, :]))
        #print(df_concat.index) 
        df_concat = df_concat.drop(index='{}'.format(uniprot_id01+'+'))
        print("i: {}".format(i))
        # df_concat.to_csv('./outputdir_rf_cls_priority/{}_{}/df_concat_{}.csv'.format(i, protease, protease))
        #print(df_concat.index) 
        #print(df_concat)
        #print(df_concat.shape)

        df_concat_pb_seq = df_concat_pb_seq.drop(index='{}'.format(uniprot_id01+'+'))
        # df_concat_pb_seq.to_csv('./outputdir_rf_cls_priority/{}_{}/df_concat_pb_seq_{}.csv'.format(i, protease, protease))
        #print("df_concat_pb_seq.")
        #print(df_concat_pb_seq.index) 
        #print(df_concat_pb_seq)
        #print(df_concat_pb_seq.shape)
        #df_concat_pb_go = df_concat_pb_go.drop(index='{}'.format(uniprot_id01+'+'))



        
        # create negative data
        df_concat_negative = pd.DataFrame([test_array], index=['{}'.format(uniprot_id01+'+')])
        df_concat_negative['label'] = 0 
        #print("df_concat_negative.")
        #print(df_concat_negative)
        #print(df_concat_negative.shape)

        test_array_pb_seq = local_representations_negative[0].flatten()
        df_concat_pb_seq_negative = pd.DataFrame([test_array_pb_seq], index=['{}'.format(uniprot_id01+'+')])
        df_concat_pb_seq_negative['label'] = 0

        # test_array_pb_go = global_representations_negative[0].flatten()
        #df_concat_pb_go_negative = pd.DataFrame([test_array_pb_go], index=['{}'.format(uniprot_id01+'+')])
        #df_concat_pb_go_negative['label'] = 0

        for j in range(len(df_negative)):
        #for j in range(10):
            print("="*10 + "negative encoding {}/{}".format(j+1, len(df_negative)) + "="*10)
            #print("i in negative.: {}".format(i))
            #print(df_negative)
            # uniprot_id = df_negative.iloc[i, 1]
            uniprot_id = df_negative['uniprot_id'][j]
            #uniprot_id = df_negative['uniprot_id'][i]
            
            print("uniprot_id: {}".format(uniprot_id))
            # cleave_pattern = df_negative.iloc[i, 3]
            negative_pattern = df_negative['negative_pattern'][j]
            print("len of negative pattern seq: {}".format(len(negative_pattern)))

            #cleave_pattern = df_negative.iloc['cleave_pattern'][i]
            print("cleave_pattern for df_concat_negative.: {}".format(negative_pattern))
            one_hot_seq = char2vec(negative_pattern, n_mer)
            one_hot_array = one_hot_seq.to_numpy().flatten()
            pb_seq_array =  local_representations_negative[j].flatten()
            print("pb_seq_array in negative : {}".format(pb_seq_array.shape))
            # pb_go_array =  global_representations_negative[j].flatten()
            
            df = pd.DataFrame([one_hot_array], index=['{}'.format(uniprot_id)])
            df['label'] = 0   
            df_pb_seq = pd.DataFrame([pb_seq_array], index=['{}'.format(uniprot_id)])
            df_pb_seq['label'] = 0
            # df_pb_go = pd.DataFrame([pb_go_array], index=['{}'.format(uniprot_id)])
            # df_pb_go['label'] = 0

            #print("df for df_concat.")
            #print(df)
            #df = pd.DataFrame([one_hot_seq], index=['{}'.format(uniprot_id)])
            #df['label'] = 1
            df_concat_pb_seq_negative = pd.concat([df_concat_pb_seq_negative, df_pb_seq])
            # df_concat_pb_go_negative = pd.concat([df_concat_pb_go_negative, df_pb_go])
            #print("df for df_concat.")
            #print(df)
            df_concat_negative = pd.concat([df_concat_negative, df])
        
        
        df_concat_negative = df_concat_negative.drop(index='{}'.format(uniprot_id01+'+'))
        df_concat_pb_seq_negative = df_concat_pb_seq_negative.drop(index='{}'.format(uniprot_id01+'+'))
        #df_concat_pb_go_negative = df_concat_pb_go_negative.drop(index='{}'.format(uniprot_id01+'+'))
        
        # device = set_device()
        data_columns_positive = len(df_concat_pb_seq.columns)-1
        x_data_positive = df_concat_pb_seq.iloc[:, 0:data_columns_positive]
        y_data_positive = df_concat_pb_seq.iloc[:, data_columns_positive]

        data_columns_negative = len(df_concat_pb_seq_negative.columns)-1
        x_data_negative = df_concat_pb_seq_negative.iloc[:, 0:data_columns_negative]
        y_data_negative = df_concat_pb_seq_negative.iloc[:, data_columns_negative]
        
        
        # ============================================================
        # Use the pre-split CSV files in S01-247_dataset_3CV_withSpro.
        #   1) fold1-3 train/val CSVs: fixed 3-fold cross validation
        #   2) S01.247.train.csv / S01.247.test.csv: final held-out test
        #   3) S01.247.dataset.csv: train the final model for Spike prediction
        # ============================================================
        seed = 42
        split_dataset_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            #"S01-247_dataset_3CV_withSpro"
            #"S08071_3CV_withSproAndHemagglutinin_DefinitiveEdition"
            "S08071_withSpro_DefinitiveEdition"
        )
        print("split_dataset_dir:", split_dataset_dir)
        if not os.path.isdir(split_dataset_dir):
            raise FileNotFoundError(
                "Pre-split dataset directory was not found: {}".format(split_dataset_dir)
            )

        def _normalize_seq(seq):
            # Keep the 160-mer positional frame. A '-' means an out-of-range/gap
            # position (e.g. before the N-terminus), so do NOT delete it.
            # ProteinBERT cannot consume '-' directly; represent it as X instead.
            return str(seq).strip().upper().replace("-", "X")

        def _find_column(df, candidates):
            lower_to_original = {str(c).strip().lower(): c for c in df.columns}
            for candidate in candidates:
                if candidate.lower() in lower_to_original:
                    return lower_to_original[candidate.lower()]
            return None

        def _labels_to_binary(series, csv_path):
            if pd.api.types.is_numeric_dtype(series):
                labels = series.astype(int).to_numpy()
            else:
                mapping = {
                    '1': 1, '0': 0,
                    'positive': 1, 'pos': 1, 'posi': 1, '+': 1,
                    'negative': 0, 'neg': 0, 'nega': 0, '-': 0
                }
                converted = []
                for value in series:
                    key = str(value).strip().lower()
                    if key not in mapping:
                        raise ValueError(
                            "Unsupported label {!r} in {}".format(value, csv_path)
                        )
                    converted.append(mapping[key])
                labels = np.asarray(converted, dtype=int)

            unknown = set(np.unique(labels).tolist()) - {0, 1}
            if unknown:
                raise ValueError(
                    "Labels must be binary 0/1 in {}. Found: {}".format(csv_path, sorted(unknown))
                )
            return labels

        # Each pre-split CSV is the source of truth.
        # Encode its sequence column directly with ProteinBERT instead of trying
        # to match the row back to df_positive/df_negative. This is important
        # because S01-247_dataset_3CV_withSpro can contain rows that are not in
        # the older ./proteases positive/negative source files.
        presplit_model_generator, presplit_input_encoder = load_pretrained_model()
        presplit_model_cache = {}
        presplit_feature_cache = {}

        def _encode_sequences_with_proteinbert(sequence_rows, csv_path):
            if len(sequence_rows) == 0:
                raise ValueError("No sequences found in {}".format(csv_path))

            normalized = [_normalize_seq(seq) for seq in sequence_rows]

            # This analysis uses a fixed 160-mer positional window. Replacing '-'
            # with X preserves that frame, so every normalized sequence must stay
            # exactly 160 characters long and every ProteinBERT model input is 162
            # tokens (160 residues + the model's two special positions).
            target_raw_len = n_mer
            bad_lengths = [
                (row_no, len(seq), seq)
                for row_no, seq in enumerate(normalized)
                if len(seq) != target_raw_len
            ]
            if bad_lengths:
                raise ValueError(
                    "Sequence length mismatch after gap->X conversion in {}. "
                    "Expected {} characters. First mismatches: {}".format(
                        csv_path, target_raw_len, bad_lengths[:10]
                    )
                )

            # Fail early with a clear message if this installed ProteinBERT version
            # does not support X. (The project's BA.2 sequence already contains X,
            # so X support is expected in the environment used for this script.)
            try:
                tokenize_seq("X")
            except Exception as exc:
                raise RuntimeError(
                    "This ProteinBERT tokenizer does not accept 'X'. "
                    "Gap positions are converted from '-' to 'X' to preserve the "
                    "160-mer coordinates, so please use a ProteinBERT version whose "
                    "tokenizer supports X. Original error: {}".format(exc)
                ) from exc

            invalid = []
            for row_no, seq in enumerate(normalized):
                if not seq:
                    invalid.append((row_no, 'empty sequence'))
                    continue
                bad_chars = sorted(set(seq) - set(AMINO_ACIDS))
                if bad_chars:
                    invalid.append((row_no, 'invalid amino acid(s): {}'.format(''.join(bad_chars))))

            if invalid:
                raise ValueError(
                    "Invalid sequence(s) in {}. First problems: {}".format(
                        csv_path, invalid[:10]
                    )
                )

            # ProteinBERT models are length-specific. Encode only sequences that
            # have not already appeared in an earlier fold/file, then reuse the
            # cached embedding. This avoids repeating the expensive ProteinBERT
            # forward pass for the same sequence across fold1-3/train/test/dataset.
            unseen_by_length = {}
            for seq in normalized:
                if seq not in presplit_feature_cache:
                    unseen_by_length.setdefault(len(seq), [])
                    if seq not in unseen_by_length[len(seq)]:
                        unseen_by_length[len(seq)].append(seq)

            for raw_len, seq_list in sorted(unseen_by_length.items()):
                seq_len = raw_len + 2
                if seq_len not in presplit_model_cache:
                    print("Create ProteinBERT model for pre-split CSV seq_len={}".format(seq_len))
                    presplit_model_cache[seq_len] = get_model_with_hidden_layers_as_outputs(
                        presplit_model_generator.create_model(seq_len)
                    )

                model_for_len = presplit_model_cache[seq_len]
                seq_series = pd.Series(seq_list, dtype='object')
                encoded_seq, encoded_annt = presplit_input_encoder.encode_X(seq_series, seq_len)

                # Predict one row at a time to keep memory consumption low,
                # consistent with the positive/negative encoding above.
                for local_no, seq in enumerate(seq_list):
                    local_rep, _ = model_for_len.predict(
                        [encoded_seq[local_no:local_no+1], encoded_annt[local_no:local_no+1]],
                        batch_size=1,
                        verbose=0
                    )
                    presplit_feature_cache[seq] = local_rep[0].astype(np.float32).flatten()

            features = [presplit_feature_cache[seq] for seq in normalized]
            x = np.vstack(features).astype(np.float32)
            return x, normalized

        def load_presplit_csv(csv_path):
            if not os.path.isfile(csv_path):
                raise FileNotFoundError("Pre-split CSV was not found: {}".format(csv_path))

            df_split = pd.read_csv(csv_path)
            # Ignore an automatically written pandas index column.
            df_split = df_split.loc[:, ~df_split.columns.astype(str).str.startswith('Unnamed:')]

            label_col = _find_column(
                df_split,
                ['label', 'y', 'target', 'class', 'y_train', 'y_val', 'y_test']
            )
            seq_col = _find_column(
                df_split,
                ['sequence', 'seq', 'pattern', 'cleave_pattern', 'negative_pattern',
                 'x', 'x_train', 'x_val', 'X_val', 'x_test', 'test_pattern']
            )

            if label_col is None:
                # Fallback: find a binary-looking column.
                for col in df_split.columns:
                    values = pd.to_numeric(df_split[col], errors='coerce').dropna().unique()
                    if len(values) > 0 and set(values.tolist()).issubset({0, 1}):
                        label_col = col
                        break

            if label_col is None:
                raise ValueError(
                    "Could not detect the label column in {}. columns={}".format(
                        csv_path, list(df_split.columns)
                    )
                )
            if seq_col is None:
                raise ValueError(
                    "Could not detect the sequence column in {}. columns={}".format(
                        csv_path, list(df_split.columns)
                    )
                )

            y = _labels_to_binary(df_split[label_col], csv_path)
            sequence_rows = df_split[seq_col].astype(str).tolist()
            x, normalized_sequences = _encode_sequences_with_proteinbert(sequence_rows, csv_path)

            if len(x) != len(y):
                raise ValueError(
                    "Feature/label row mismatch in {}: X={}, y={}".format(csv_path, len(x), len(y))
                )

            print(
                "loaded+encoded:", os.path.basename(csv_path),
                "X:", x.shape, "y:", y.shape,
                "class counts:", dict(zip(*np.unique(y, return_counts=True)))
            )
            return x, y, normalized_sequences, df_split

        def new_forest():
            return RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
            )

        # -------------------- fixed 3-fold cross validation --------------------
        roc_cv_list = []
        cv_optimal_thresholds = []

        for fold_no in range(1, k_cv_num + 1):
            print("#"*10 + " fixed 3-CV fold {} ".format(fold_no) + "#"*10)
            train_csv = os.path.join(
                split_dataset_dir, "{}.fold{}.train.csv".format(protease, fold_no)
            )
            val_csv = os.path.join(
                split_dataset_dir, "{}.fold{}.val.csv".format(protease, fold_no)
            )

            x_train, y_train, x_train_seq_list, _ = load_presplit_csv(train_csv)
            x_val, y_val, x_val_seq_list, _ = load_presplit_csv(val_csv)

            # Keep trace CSVs in the existing output directory.
            pd.DataFrame({
                'x_train': x_train_seq_list,
                'y_train': y_train
            }).to_csv(
                "./outputdir_rf_cls_priority/pbseq/{}_{}/train-data{}_cv{}.csv".format(
                    i, protease, protease, fold_no
                ), index=False
            )
            pd.DataFrame({
                'X_val': x_val_seq_list,
                'y_val': y_val
            }).to_csv(
                "./outputdir_rf_cls_priority/pbseq/{}_{}/val-data{}_cv{}.csv".format(
                    i, protease, protease, fold_no
                ), index=False
            )

            forest_cv = new_forest()
            forest_cv.fit(x_train, y_train)
            y_true = y_val
            y_score = forest_cv.predict_proba(x_val)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc_value = roc_auc_score(y_true, y_score)
            roc_cv_list.append(roc_auc_value)

            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = float(thresholds[optimal_idx])
            cv_optimal_thresholds.append(optimal_threshold)

            pd.DataFrame({
                'optimal_threshold': [optimal_threshold]
            }).to_csv(
                "./outputdir_rf_cls_priority/pbseq/{}_{}/df_optimal_threshold_{}_cv{}.csv".format(
                    i, protease, protease, fold_no
                ), index=False
            )

            y_pred = (y_score >= optimal_threshold).astype(int)
            f1_value = f1_score(y_true, y_pred)
            precision_value = precision_score(y_true, y_pred, zero_division=0)
            recall_value = recall_score(y_true, y_pred, zero_division=0)
            accuracy_value = accuracy_score(y_true, y_pred)

            pd.DataFrame({
                'roc_auc': [roc_auc_value],
                'f1_score': [f1_value],
                'precision_score': [precision_value],
                'recall_score': [recall_value],
                'accuracy_value': [accuracy_value]
            }).to_csv(
                './outputdir_rf_cls_priority/pbseq/{}_{}/roc_auc_f1score_{}_cv{}.csv'.format(
                    i, protease, protease, fold_no
                ), index=False
            )

            plt.plot(fpr, tpr, marker='o')
            plt.xlabel('FPR: False positive rate')
            plt.ylabel('TPR: True positive rate')
            plt.title('ROC_{}_CV{}'.format(protease, fold_no))
            plt.grid()
            plt.savefig(
                './outputdir_rf_cls_priority/pbseq/{}_{}/roc_curve_{}_cv{}.png'.format(
                    i, protease, protease, fold_no
                )
            )
            plt.clf()
            plt.close()

            pd.DataFrame({
                'y_val': y_val,
                'predict_prob_label1_val': y_score
            }).to_csv(
                "./outputdir_rf_cls_priority/pbseq/{}_{}/df_val_predicteds_{}_cv{}.csv".format(
                    i, protease, protease, fold_no
                ), index=False
            )

            # Probability-score errors (for binary labels, MSE is the Brier score).
            mae = mean_absolute_error(y_true, y_score)
            mse = mean_squared_error(y_true, y_score)
            pd.DataFrame([[mae, mse]], columns=["val_mae", "val_mse"]).to_csv(
                "./outputdir_rf_cls_priority/pbseq/{}_{}/df_val_error_{}_cv{}.csv".format(
                    i, protease, protease, fold_no
                ), index=False
            )

        pd.DataFrame({
            'fold': list(range(1, k_cv_num + 1)),
            'roc_auc': roc_cv_list,
            'optimal_threshold': cv_optimal_thresholds
        }).to_csv(
            './outputdir_rf_cls_priority/pbseq/{}_{}/cv_summary_{}.csv'.format(
                i, protease, protease
            ), index=False
        )

        # -------------------- held-out test evaluation --------------------
        print("#"*10 + " Start fixed test validation " + "#"*10)
        final_train_csv = os.path.join(split_dataset_dir, "{}.train.csv".format(protease))
        final_test_csv = os.path.join(split_dataset_dir, "{}.test.csv".format(protease))
        x_learn, y_learn, x_learn_seq_list, _ = load_presplit_csv(final_train_csv)
        x_test, y_test, x_test_seq_list, _ = load_presplit_csv(final_test_csv)

        pd.DataFrame({
            'x_learn': x_learn_seq_list,
            'y_learn': y_learn
        }).to_csv(
            "./outputdir_rf_cls_priority/pbseq/{}_{}/learn-data_{}_seqs.csv".format(
                i, protease, protease
            ), index=False
        )
        pd.DataFrame({
            'x_test': x_test_seq_list,
            'y_test': y_test
        }).to_csv(
            "./outputdir_rf_cls_priority/pbseq/{}_{}/test-data_{}_seqs.csv".format(
                i, protease, protease
            ), index=False
        )

        forest = new_forest()
        forest.fit(x_learn, y_learn)
        y_score = forest.predict_proba(x_test)[:, 1]
        y_true = y_test

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc_value = roc_auc_score(y_true, y_score)

        # Do not optimize the classification threshold on the held-out test set.
        # Use the mean threshold obtained only from the three validation folds.
        test_threshold = float(np.mean(cv_optimal_thresholds))
        pd.DataFrame({
            'optimal_threshold': [test_threshold],
            'threshold_source': ['mean_of_3CV_validation_thresholds']
        }).to_csv(
            "./outputdir_rf_cls_priority/pbseq/{}_{}/df_optimal_threshold_{}_test.csv".format(
                i, protease, protease
            ), index=False
        )
        y_pred = (y_score >= test_threshold).astype(int)

        f1_value = f1_score(y_true, y_pred)
        precision_value = precision_score(y_true, y_pred, zero_division=0)
        recall_value = recall_score(y_true, y_pred, zero_division=0)
        accuracy_value = accuracy_score(y_true, y_pred)

        pd.DataFrame({
            'roc_auc': [roc_auc_value],
            'f1_score': [f1_value],
            'precision_score': [precision_value],
            'recall_score': [recall_value],
            'accuracy_value': [accuracy_value],
            'classification_threshold': [test_threshold]
        }).to_csv(
            './outputdir_rf_cls_priority/pbseq/{}_{}/roc_auc_f1score_{}_test.csv'.format(
                i, protease, protease
            ), index=False
        )

        plt.plot(fpr, tpr, marker='o')
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.title('ROC_{}_test'.format(protease))
        plt.grid()
        plt.savefig(
            './outputdir_rf_cls_priority/pbseq/{}_{}/roc_curve_{}_test.png'.format(
                i, protease, protease
            )
        )
        plt.clf()
        plt.close()

        pd.DataFrame({
            'y_test': y_test,
            'predict_prob_label1_test': y_score,
            'predicted_class': y_pred
        }).to_csv(
            "./outputdir_rf_cls_priority/pbseq/{}_{}/df_test_predicteds_{}_test.csv".format(
                i, protease, protease
            ), index=False
        )

        # Probability-score errors (for binary labels, MSE is the Brier score).
        mae = mean_absolute_error(y_true, y_score)
        mse = mean_squared_error(y_true, y_score)
        pd.DataFrame([[mae, mse]], columns=["test_mae", "test_mse"]).to_csv(
            "./outputdir_rf_cls_priority/pbseq/{}_{}/df_val_error_{}_test.csv".format(
                i, protease, protease
            ), index=False
        )

        # -------------------- final model for Spike-protein prediction --------------------
        # Train on every row in S01.247.dataset.csv, exactly as requested.
        full_dataset_csv = os.path.join(split_dataset_dir, "{}.dataset.csv".format(protease))
        x_dataset, y_dataset, _, _ = load_presplit_csv(full_dataset_csv)
        forest = new_forest()
        forest.fit(x_dataset, y_dataset)
        print("Final model trained with:", os.path.basename(full_dataset_csv), x_dataset.shape)

        pretrained_model_generator, input_encoder = load_pretrained_model()
        origin_fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
        for j in range(8):
            if j == 0:
                fullseq = origin_fullseq
                mutation_name = "origin"
            elif j == 1:
                fullseq = test_data_gen_alpha()
                mutation_name = "alpha"
            elif j == 2:
                fullseq = test_data_gen_beta()
                mutation_name = "beta"
            elif j == 3:
                fullseq = test_data_gen_gamma()
                mutation_name = "gamma"
            elif j == 4:
                fullseq = test_data_gen_delta()
                mutation_name = "delta"
            elif j == 5:
                fullseq = test_data_gen_omicron()
                mutation_name = "omicron"
            elif j == 6:
                fullseq = test_data_gen_omicron_ba1()
                mutation_name = "omicron_ba1"
            elif j == 7:
                fullseq = test_data_gen_omicron_ba2()
                mutation_name = "omicron_ba2"
            """
            elif j == 7:
                fullseq = test_data_gen_omicron_ba3()
                mutation_name = "omicron_ba3"
            elif j == 8:
                fullseq = test_data_gen_omicron_jn1()
                mutation_name = "omicron_jn1"
            elif j == 9:
                fullseq = test_data_gen_omicron_eg5()
                mutation_name = "omicron_eg5"
            elif j == 10:
                fullseq = test_data_gen_omicron_kp3()
                mutation_name = "omicron_kp3"
            elif j == 11:
                fullseq = test_data_gen_omicron_xec()
                mutation_name = "omicron_xec"
            else:
                pass
            """

            spike_protein_eight_aa_list = test_data_gen(fullseq, trim_num)
            df_test_pattern = pd.DataFrame(spike_protein_eight_aa_list, columns=["test_pattern"], index=list(range(len(spike_protein_eight_aa_list))))
            seqs = df_test_pattern["test_pattern"]
            
            seq_len= len(seqs[0])+2

            model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))
            encoded_x = input_encoder.encode_X(seqs, seq_len)
            # local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
            #local_representations = model.predict(encoded_x, batch_size=batch_size)
            #local_representations = np.array(local_representations)

            predicted_outputs = model.predict(encoded_x, batch_size=batch_size)
            #predicted_outputs = model.predict(encoded_x)

            if isinstance(predicted_outputs, (list, tuple)) and len(predicted_outputs) == 2:
                local_representations = np.asarray(predicted_outputs[0])
                global_representations = np.asarray(predicted_outputs[1])
            else:
                raise ValueError(
                    "Expected two ProteinBERT outputs, but got: "
                    f"type={type(predicted_outputs)}"
                )
            print("local_representations.shape:", local_representations.shape)
            print("global_representations.shape:", global_representations.shape)
            print("type(local_representations):", type(local_representations))
            if isinstance(local_representations, (list, tuple)):
                print("len(local_representations):", len(local_representations))
                for i, value in enumerate(local_representations):
                    print(f"output[{i}] type:", type(value))
                    try:
                        print(f"output[{i}] shape:", np.asarray(value).shape)
                    except Exception as e:
                        print(f"output[{i}] shape error:", e)
            local_representations = np.array(local_representations)

            x_test = local_representations.reshape(-1, int(local_representations.shape[1]*local_representations.shape[2]))
            y_test = np.zeros((len(x_test), 1))

            #test_dataloader = test_data_loader(x_test, y_test)

            #d_input = data_columns_positive
            #_output = 1 #2. this parameter is used in Decoder.
            #d_model = 64 #64.
            #nhead = 8 #2. 4. 
            #dim_feedforward = 1024   #2024. 16. 88.
            #num_encoder_layers = 1 
            #num_decoder_layers = 0
            #dropout = 0.01  #0.02. 0.1, 
            #src_len = 18 # it was 18.
            #tgt_len = 2 # it was 2. 
            #batch_size = 8
            #epochs = 5 #30, 200
            
            #best_loss = float('Inf')
            #best_model = None
            #model = Transformer(num_encoder_layers=num_encoder_layers,
            #                    num_decoder_layers=num_decoder_layers,
            #                    d_model=d_model,
            #                   d_input=d_input, 
            #                   d_output=d_output,
            #                   dim_feedforward=dim_feedforward,
            #                    dropout=dropout, nhead=nhead
            #                ).to(device)
            #model = NN(d_model, d_input, d_output) 
            #predictdata = predict(test_dataloader, model, device, model_path)
            # RandomForestClassifier: class-1 probability is used as the continuous cleavage score.
            predictdata = forest.predict_proba(x_test)[:, 1]
            predictclass = forest.predict(x_test)

            print("predictdata = P(label=1) in main function.")
            print(predictdata.shape)
            df_predict = pd.DataFrame({
                "predict": predictdata,
                "predict_label": predictclass.astype(int),
            }, index=list(range(len(predictdata))))
            df_predict.to_csv("./outputdir_rf_cls_priority/pbseq/{}_{}/predict_sprotein_{}.csv".format(i, protease, protease))
            df_concat_predict_seqs = pd.concat([df_predict, seqs], axis=1)
            df_concat_predict_seqs.to_csv("./outputdir_rf_cls_priority/pbseq/{}_{}/predict_sprotein_seqs_{}_{}.csv".format(i, protease, protease, mutation_name))

"""
def con_db():
    # コネクションの作成
    conn = mydb.connect(
        host='localhost',
        port=3306,
        user='root',
        password='miyazakilab',
        #password='KtHk23#8', 
        #database='meropsrefs01'
        database='meropsweb12_1'
        #database='meropsweb121'
    )

    # DB操作用にカーソルを作成
    cur = conn.cursor()
    return cur
"""
#def con_db():
#    # コネクションの作成
#    conn = mydb.connect(
#        host='localhost',
#        port=3306,
#        #user='root',
#        user="meropsuser",
#        #password='miyazakilab',
#        #password='KtHk23#8', 
#        #password='QuNr#he',
#        #password='QmCy94#6', 
#        password='WrXu#857', 
#        #database='meropsrefs01'
#        #database='meropsweb12_1'
#        database='meropsweb121'
#   )
#    # DB操作用にカーソルを作成
#    cur = conn.cursor()
#    return cur
def con_db():
    # コネクションの作成
    conn = mydb.connect(
        host='localhost',
        port=3306,
        user='root',
        #password='miyazakilab',
        #password='KtHk23#8', 
        password='QuNr#he',
        #database='meropsrefs01'
        #database='meropsweb12_1'
        database='meropsweb121'
    )

    # DB操作用にカーソルを作成
    cur = conn.cursor()
    return cur

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # 最小値, 最大値からエリアの領域を割り出す
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # resolutionの間隔で区切った領域を定義
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    # print(xx1.shape)

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

   # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
            alpha=1.0, linewidth=1, marker='o',
            s=55, label='test set')

def tensor_to_list(tensor_list):
    concat_list = []
    for i in range(len(tensor_list)):
        temp = tensor_list[i].tolist()
        concat_list.append(temp)
    count = 0
    for item in concat_list:
        count += len(item)
    return list(itertools.chain.from_iterable(concat_list))

def create_merops_code_table():
    csv_file = open("./testdata/merops_code_mece.csv", "r", encoding="ms932", errors="", newline="" )
    #リスト形式
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    #print(f)
    #print(len(f))

    i = 0
    for row in f:
        print(row)
        merops_code_mece = row
        i = i + 1
        if i == 1:
            break

    #print("merops_code_mece")
    #print(merops_code_mece)
    #len(merops_code_mece)
    return merops_code_mece






arr = [[0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0]
       ]
df = pd.DataFrame(arr,
                  columns = ["P4","P3","P2", "P1", "P1dash", "P2dash", "P3dash", "P4dash"],
                  index = ["A","C","D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])

def char2vec(cleave_pattern, n_mer):
    #arr = [[0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0],
    #    [0,0,0,0,0,0,0,0]
    #    ]
    arr = [[0]*n_mer]*20
    col_num = int(n_mer//2)
    columns_list = []
    columns_p = []
    columns_p_dash = []
    for i in range(col_num):
        columns_p.append("P{}".format(col_num - i))
        columns_p_dash.append("P{}dash".format(i + 1))
    columns_list = columns_p + columns_p_dash

    df = pd.DataFrame(arr,
                    #columns = ["P4","P3","P2", "P1", "P1dash", "P2dash", "P3dash", "P4dash"],
                    columns = columns_list,
                    index = ["A","C","D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])
    print("cleave_pattern: ")
    print(cleave_pattern)

    for i in range(len(cleave_pattern)): 
        #print("j in char2vec func.: {}".format(j))
        aa = cleave_pattern[i]  
        if aa == "A":
            #x_train[num][0][j] += 1.0
            df.iloc[0, i] +=1.0
        elif aa == "C":
            #x_train[num][1][j] += 1.0 
            df.iloc[1, i] +=1.0
        elif aa == "D":
            #x_train[num][2][j] += 1.0 
            df.iloc[2, i] +=1.0
        elif aa == "E":
            #x_train[num][3][j] += 1.0 
            df.iloc[3, i] +=1.0
        elif aa == "F":
            #x_train[num][4][j] += 1.0
            df.iloc[4, i] +=1.0
        elif aa == "G":
            #x_train[num][5][j] += 1.0
            df.iloc[5, i] +=1.0
        elif aa == "H":
            #x_train[num][6][j] += 1.0
            df.iloc[6, i] +=1.0
        elif aa == "I":
            #x_train[num][7][j] += 1.0
            df.iloc[7, i] +=1.0
        elif aa == "K":
            #x_train[num][8][j] += 1.0
            df.iloc[8, i] +=1.0
        elif aa == "L":
            #x_train[num][9][j] += 1.0
            df.iloc[9, i] +=1.0
        elif aa == "M":
            #x_train[num][10][j] += 1.0
            df.iloc[10, i] +=1.0
        elif aa == "N":
            #x_train[num][11][j] += 1.0
            df.iloc[11, i] +=1.0
        elif aa == "P":
            #x_train[num][12][j] += 1.0
            df.iloc[12, i] +=1.0
        elif aa == "Q":
            #x_train[num][13][j] += 1.0
            df.iloc[13, i] +=1.0
        elif aa == "R":
            #x_train[num][14][j] += 1.0
            df.iloc[14, i] +=1.0
        elif aa == "S":
            #x_train[num][15][j] += 1.0
            df.iloc[15, i] +=1.0
        elif aa == "T":
            #x_train[num][16][j] += 1.0
            df.iloc[16, i] +=1.0
        elif aa == "V":
            #x_train[num][17][j] += 1.0
            df.iloc[17, i] +=1.0
        elif aa == "W":
            #x_train[num][18][j] += 1.0
            df.iloc[18, i] +=1.0
        elif aa == "Y":
            #x_train[num][19][j] += 1.0
            df.iloc[19, i] +=1.0
        else:
            pass
    return df

"""
def set_device():
    device_var = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用デバイス:", device_var)
    device = torch.device(device_var)
    return device
"""


def set_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("使用デバイス: GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return gpus
    else:
        print("使用デバイス: CPU")
        return gpus

#def set_data_param():
#    #データ前処理 transform を設定
#    transform = transforms.Compose(
#    [transforms.ToTensor(),                      # Tensor変換とshape変換 [H, W, C] -> [C, H, W]
#        transforms.Normalize((0.5, ), (0.5, ))])    # 標準化 平均:0.5  標準偏差:0.5
#   return 1
def aaseq_from_uid(uid, protease_turn, substrate_turn):
    df = pd.DataFrame(np.arange(3).reshape(1, 3), columns=['uniprotKB_accession', 'function', 'sequence'], index=['protease'+str(protease_turn)+'_substrate'+str(substrate_turn)])

    for column_name in df:
        df[column_name] = df[column_name].astype(str)

    df['uniprotKB_accession'][0] = uid

    #display(df)    

    url = "https://www.uniprot.org/uniprot/" + uid + ".xml"
    f = urlopen(url)
    xml = f.read()
    root = etree.fromstring(xml)
    
    #以下のコードは下の説明を参照
    function = root.find('./entry/comment[@type="function"]', root.nsmap)
    if function==None:
        print("function was not detected.")
        pass
    else:
        df["function"][0] = function[0].text
    sequence = root.find('./entry/sequence', root.nsmap) 
    if sequence==None: 
        print("sequence was not detected.")
        pass 
    else: 
        df["sequence"][0] = sequence.text 
    print("df: ")
    print(df)
    print(df["sequence"][0]) 
    return df["sequence"][0]

def test_data_gen(fullseq, trim_num):
    #uniprot_id = "P0DTC2"
    #fullseq = aaseq_from_uid(uniprot_id, 0, 0)
    # 1273 aa
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_045512.2
    #fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    #fullseq_list = list(fullseq)


    spike_protein_aa_list = []

    for loc_ in range(len(fullseq)-trim_num):
        eight_seq = fullseq[loc_: loc_+trim_num]
        spike_protein_aa_list.append(eight_seq)

    return spike_protein_aa_list

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

def test_data_gen_omicron_ba3():
    fullseq_mutation = ""
    return fullseq_mutation

def test_data_gen_omicron_jn1():
    # S	changes	T19I, R21T, A27S, S50L, V127F, G142D, F157S, R158G, L212I, V213G, L216F, H245N, A264D, I332V, G339H, K356T, S371F, S373P, S375F, T376A, R403K, D405N, R408S, K417N, N440K, V445H, G446S, N450D, L452W, L455S, N460K, S477N, T478K, N481K, E484K, F486P, Q498R, N501Y, Y505H, E554K, A570V, D614G, P621S, H655Y, N679K, P681R, N764K, D796Y, S939F, Q954H, N969K, P1143L
    # S	reversionsToRoot	A67A, T95T, V143V, Y145Y, Q493Q
    # S	gaps	L24-, P25-, P26-, H69-, V70-, Y144-, N211-, V483-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_jn1.csv', header=None)
    #print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[2, 1:9].tolist()
    #print(content_list02)
    for content in content_list:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            #print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            #print("new_aa: {}".format(new_aa))
            #print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            #print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_eg5():
    # S	changes	T19I, A27S, V83A, G142D, H146Q, Q183E, V213E, G252V, G339H, R346T, L368I, S371F, S373P, S375F, T376A, D405N, R408S, K417N, N440K, V445P, G446S, F456L, N460K, S477N, T478K, E484A, F486P, F490S, Q498R, N501Y, Y505H, D614G, S640F, H655Y, N679K, P681H, N764K, D796Y, Q954H, N969K
    # S	gaps	L24-, P25-, P26-, Y144-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_eg5.csv', header=None)
    #print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[1, 1:5].tolist()
    #print(content_list02)
    for content in content_list:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            #print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            #print("new_aa: {}".format(new_aa))
            #print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            #print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_kp3():
    # S	changes	T19I, R21T, A27S, S50L, V127F, G142D, F157S, R158G, L212I, V213G, L216F, H245N, A264D, I332V, G339H, K356T, S371F, S373P, S375F, T376A, R403K, D405N, R408S, K417N, N440K, V445H, G446S, N450D, L452W, L455S, F456L, N460K, S477N, T478K, N481K, E484K, F486P, Q493E, Q498R, N501Y, Y505H, E554K, A570V, D614G, P621S, H655Y, N679K, P681R, N764K, D796Y, S939F, Q954H, N969K, V1104L, P1143L
    # S	reversionsToRoot	A67A, T95T, V143V, Y145Y
    # S	gaps	L24-, P25-, P26-, H69-, V70-, Y144-, N211-, V483-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_kp3.csv', header=None)
    #print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[2, 1:9].tolist()
    #print(content_list02)
    for content in content_list:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            #print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            #print("new_aa: {}".format(new_aa))
            #print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            #print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

def test_data_gen_omicron_xec():
    # S	changes	T19I, R21T, T22N, A27S, S50L, F59S, V127F, G142D, F157S, R158G, L212I, V213G, L216F, H245N, A264D, I332V, G339H, K356T, S371F, S373P, S375F, T376A, R403K, D405N, R408S, K417N, N440K, V445H, G446S, N450D, L452W, L455S, F456L, N460K, S477N, T478K, N481K, E484K, F486P, Q493E, Q498R, N501Y, Y505H, E554K, A570V, D614G, P621S, H655Y, N679K, P681R, N764K, D796Y, S939F, Q954H, N969K, V1104L, P1143L
    # S	gaps	L24-, P25-, P26-, H69-, V70-, Y144-, N211-, V483-
    fullseq = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    fullseq_list = list(fullseq)
    df = pd.read_csv('./mutations/omicron_xec.csv', header=None)
    #print(df)
    #content = r'hellow python, 123,end' 
    content_list = df.iloc[0, 1:].tolist()
    #print(content_list)
    content_list02 = df.iloc[1, 1:9].tolist()
    #print(content_list02)
    for content in content_list:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)([A-Z])$'
        result = re.match(pattern, content)
        if result: #none以外の場合
            mutation_position = int(result.group(2))
            #print("metation_position: {}".format(mutation_position))
            new_aa = result.group(3)
            #print("new_aa: {}".format(new_aa))
            #print("fullseq_list[mutation_position-1] : {}".format(fullseq_list[mutation_position-1] ))
            fullseq_list[mutation_position-1] = new_aa
    for content in content_list02:
        #print("="*10)
        #print("content: {}".format(content))
        pattern = r'^\s*([A-Z])(\d+)(.+)$'
        result = re.match(pattern, content) 
        gap_count = 0
        if result: #none以外の場合
            gap_position = int(result.group(2))
            #print("gap_position: {}".format(gap_position))
            del fullseq_list[gap_position-gap_count-1]
            gap_count += 1
    fullseq_mutation = "".join(fullseq_list)
    return fullseq_mutation

if __name__ == '__main__':
    start = time.time()
    
    main()

    end = time.time()
    time_diff = end - start
    print(time_diff)
    print(time_diff/60)
    print(time_diff/60/60)
    print(time_diff/60/60/24)
    
    #dt_now = datetime.datetime.now()
    #print(dt_now)
    print("END.")

