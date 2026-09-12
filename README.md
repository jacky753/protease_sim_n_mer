
# short explain
protease_dataset_generator — where positive data consists of N-mer segments spanning protease cleavage sites and negative data consists of randomly selected N-mer segments from non-cleavage regions
 
# explain
Read above explain. Protease database is from MEROPS database.

# ProteinBERT + RandomForest protease classifier — SOLID分割版

アップロードされた単一Pythonスクリプトを、SOLID原則を意識して責務ごとのファイルへ分割した版です。

主となる学習処理は、元プログラムと同じ考え方を維持しています。

1. 固定済みの pre-split CSV を読み込む
2. `-` を `X` に置換し、160残基の位置フレームを維持する
3. ProteinBERT の **local representation** を取得して flatten する
4. 固定3-fold CVを実行する
5. 各validation foldで Youden J (`TPR - FPR`) により閾値を求める
6. held-out test では3つのvalidation閾値の平均を使用する
7. `<protease>.dataset.csv` 全体で最終RandomForestを学習する
8. Spike各変異株を160-mer sliding windowにして `P(label=1)` を出力する

## ファイル構成

```text
protease_rf_solid_refactor/
├─ main.py
├─ requirements.txt
├─ README.md
├─ protease_rf/
│  ├─ __init__.py
│  ├─ app.py                    # Dependency Injection / オブジェクト組み立て
│  ├─ config.py                 # 設定値のみ
│  ├─ contracts.py              # 小さなインターフェース(Protocol)
│  ├─ sequence_utils.py         # 配列正規化、label変換、window生成
│  ├─ proteinbert_features.py   # ProteinBERT特徴抽出とcache
│  ├─ dataset_loader.py         # pre-split CSV読込
│  ├─ random_forest.py          # RandomForest生成
│  ├─ evaluation.py             # ROC/AUC/F1/MAE/MSE/閾値計算
│  ├─ result_writer.py          # CSV/PNG保存だけを担当
│  ├─ training_pipeline.py      # CV → test → final training の制御
│  ├─ spike_variants.py         # Spike変異株配列
│  ├─ spike_prediction.py       # Spike sliding-window予測
│  └─ runtime.py                # TensorFlow GPU memory growth設定
└─ tests/
   ├─ test_sequence_utils.py
   └─ test_evaluation.py
```

依存関係は概ね次のようになっています。

```text
main.py
  └─ app.py  (Composition Root)
      ├─ AppConfig
      ├─ ProteinBertFeatureExtractor
      │    └─ SequenceFeatureExtractor interface
      ├─ PreSplitCsvLoader
      ├─ RandomForestFactory
      │    └─ ClassifierFactory interface
      ├─ BinaryClassificationEvaluator
      ├─ ResultWriter
      ├─ TrainingPipeline
      └─ SpikePredictionService
           └─ SpikeVariantProvider
```

## 実行前のデータ配置

デフォルトでは `main.py` があるディレクトリを `project_root` とします。
したがって、既存のデータフォルダを `main.py` と同じ階層に置いてください。

```text
protease_rf_solid_refactor/
├─ main.py
├─ S08071_withSpro_DefinitiveEdition/
│  ├─ S08.071.fold1.train.csv
│  ├─ S08.071.fold1.val.csv
│  ├─ S08.071.fold2.train.csv
│  ├─ S08.071.fold2.val.csv
│  ├─ S08.071.fold3.train.csv
│  ├─ S08.071.fold3.val.csv
│  ├─ S08.071.train.csv
│  ├─ S08.071.test.csv
│  └─ S08.071.dataset.csv
└─ protease_rf/
```

別のディレクトリを利用する場合は `main.py` で以下のように指定できます。

```python
from pathlib import Path
from protease_rf import AppConfig
from protease_rf.app import build_and_run

config = AppConfig(project_root=Path(r"C:\your\existing\project"))
build_and_run(config)
```

## 実行

```bash
python main.py
```

デフォルト設定は元プログラムに合わせて以下です。

```text
target_protease = S08.071
sequence_length = 160
cv_folds = 3
RandomForest n_estimators = 500
random_state = 42
n_jobs = -1
```

出力先も従来と同じ構造です。

```text
outputdir_rf_cls_priority/pbseq/741_S08.071/
```

## SOLID原則をどこに適用したか

### S — Single Responsibility Principle

元の `main()` はCSV読込、ProteinBERT、RandomForest、評価、ROC描画、CSV保存、Spike予測をすべて担当していました。
分割版では「変更理由」が異なる処理を別クラス/別ファイルに分けています。

例:

- CSV形式が変わる → `dataset_loader.py`
- ProteinBERTの取得方法が変わる → `proteinbert_features.py`
- RFのパラメータが変わる → `random_forest.py` / `config.py`
- 評価指標が増える → `evaluation.py`
- 出力ファイル形式が変わる → `result_writer.py`

### O — Open/Closed Principle

`TrainingPipeline` の中で `RandomForestClassifier(...)` を直接生成しません。
`ClassifierFactory` に依存させたので、将来XGBoostなどを追加するときは別Factoryを実装できます。

### L — Liskov Substitution Principle

`ProteinBertFeatureExtractor` の代わりに、同じ `SequenceFeatureExtractor` 契約を満たす ESM2 extractor 等を差し替えられる構造にしています。

### I — Interface Segregation Principle

大きな万能interfaceを作らず、

- `SequenceFeatureExtractor.encode()`
- `ClassifierFactory.create()`

という小さな契約だけを用意しています。

### D — Dependency Inversion Principle

高レベルの `TrainingPipeline` は ProteinBERT や RandomForest の具体クラスそのものではなく、必要なinterfaceに依存します。
具体クラスを組み立てる場所は `app.py` 一箇所にまとめています。

## 元プログラムから維持した重要仕様

- pre-split CSVをsource of truthとして直接ProteinBERTへ入力
- gap `-` → `X` にして160-mer座標を維持
- ProteinBERT local representationをflatten
- 同一sequenceのfeature cache
- fixed 3-fold CV
- validationでのYouden J閾値
- test dataで閾値を最適化しない
- test threshold = 3 validation thresholdsの平均
- RandomForest 500 trees / `random_state=42` / `n_jobs=-1`
- ROC PNGおよび既存CSVファイル名

また、元の `test_data_gen()` は

```python
range(len(fullseq) - trim_num)
```

だったため、理論上最後の1 windowを含みません。分割版では結果互換性を優先して
`include_last_spike_window=False` をデフォルトにしています。
全windowを使う場合だけ `True` に変更してください。

## 主処理から外した旧処理について

元 `main()` 前半には、MEROPS/MySQL参照と旧positive/negative CSVの別ProteinBERTエンコード処理がありますが、その結果の `subs`、`x_data_positive`、`x_data_negative` 等は、その後の fixed CV / held-out test / final model / Spike prediction では参照されていません。

そのため、今回のSOLID版では **学習結果を決めている実際の主経路から分離し、実行していません**。これにより不要なDB依存とProteinBERTの重複計算を避けています。

必要であれば、この旧処理も次のように独立サービスとして追加するのが適切です。

```text
MeropsRepository
LegacyPositiveNegativeDatasetEncoder
```

## テスト

ProteinBERT実データを必要としない純粋ロジックについてテストを付属しています。

```bash
python -m pytest -q
```

作成環境では4テストすべて通過しています。
