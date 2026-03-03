# Detection of Patronizng and Condescending Language

This repository contains the full implementation for the shared task on detecting **Patronising and Condescending Language (PCL)** towards vulnerable communities. The proposed approach is a contrastively fine-tuned Sentence-BERT model with hybrid lexical features and an ensemble classifier, achieving an **F1 of 0.649** on the official dev set — surpassing the RoBERTa-base baseline of 0.48.

---

## Repository Structure

```
nlp-dont-patronize-me-2026/
│
├── BestModel/                  # Trained model and all saved components
├── Predictions/                # Final dev and test set predictions
├── AblationStudies/            # Ablation study code and results
├── EDA/                        # Exploratory data analysis code and figures
└── ErrorAnalysis/              # Error analysis code and figures
```

---


## BestModel

Contains the full implementation and all saved components required to reproduce predictions.

| File | Description |
|---|---|
| [`pcl_detection_sbert.ipynb`](BestModel/pcl_detection_sbert.ipynb) | Full training pipeline — data loading, contrastive fine-tuning, feature extraction, ensemble training, and prediction |
| `finetuned_encoder.pt` | Saved weights of the contrastively fine-tuned Sentence-BERT encoder |
| `tokenizer/` | Saved tokenizer corresponding to the fine-tuned encoder |
| `classifiers.pkl` | Saved ensemble classifiers (Logistic Regression, MLP, Random Forest, Gradient Boosting) |
| `scaler.pkl` | Saved StandardScaler fit on the balanced training features |
| `tfidf.pkl` | Saved TF-IDF vectorizer fit on the training set |
| `model_config.json` | Model configuration (base model name, threshold, balance ratio, ensemble classifiers used) |
| `model_summary.txt` | Summary of dev set performance metrics |

---

## Predictions

Contains the final prediction files in the required submission format (one prediction per line: `0` = No PCL, `1` = PCL).

| File | Description |
|---|---|
| [`Predictions/dev.txt`](Predictions/dev.txt) | Predictions on the official dev set (2,094 lines) |
| [`Predictions/test.txt`](Predictions/test.txt) | Predictions on the official test set (3,832 lines) |

---

## EDA

Exploratory data analysis of the Don't Patronize Me! dataset.

| File | Description |
|---|---|
| [`pcl_eda.ipynb`](EDA/pcl_eda.ipynb) | EDA code — class distribution analysis and lexical analysis |

---

## AblationStudies

Systematic component removal experiments to quantify the contribution of each part of the pipeline.

| File | Description |
|---|---|
| [`pcl_ablation_studies.ipynb`](AblationStudies/pcl_ablation_studies.ipynb) | Ablation study code — component removal and threshold sensitivity analysis |

---

## ErrorAnalysis

Local evaluation of the best model against the vanilla SBERT baseline on the official dev set.

| File | Description |
|---|---|
| [`pcl_error_analysis.ipynb`](ErrorAnalysis/pcl_error_analysis.ipynb) | Error analysis code — confusion matrices, error categorisation, keyword-level recall, and false positive/negative inspection |


---

## Requirements

```
torch
transformers
sentence-transformers
scikit-learn
numpy
pandas
matplotlib
seaborn
tqdm
```

---

## Data

The dataset used is the [Don't Patronize Me!](https://github.com/Perez-AlmendrosC/dontpatronizeme) dataset (Pérez-Almendros et al., COLING 2020). The data files (`dontpatronizeme_pcl.tsv`, `train.csv`, `dev.csv`, `task4_test.tsv`) are not included in this repository.

---

## Reference

Pérez-Almendros, C., Espinosa-Anke, L., & Schockaert, S. (2020). *Don't Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities.* Proceedings of COLING 2020.
