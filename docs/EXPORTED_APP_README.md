---
title: Consumer Complaint Classifier
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: '3.11'
pinned: false
---

# Consumer Complaint Classifier

An automated Natural Language Processing system for financial service complaints. Enter an English customer complaint narrative to predict the appropriate department routing and view confidence probabilities across five major financial categories.

The app uses a genuinely fine-tuned DistilBERT selected using validation macro F1 before the final test phase. On the held-out 12,398-record test split, its accuracy was **85.43%** and macro F1 was **0.8467**.

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| SimpleRNN | 74.72% | 0.7263 |
| LSTM | 83.74% | 0.8231 |
| GRU | 84.14% | 0.8263 |
| DistilBERT | 85.43% | 0.8467 |

## Use and limitations

The supported labels are credit card, credit reporting, debt collection, mortgages and loans, and retail banking. The classifier is English-only and always chooses among these five categories. Unrelated inputs can still receive a category. Probabilities are uncalibrated and are not a guarantee of correctness.

Source narratives were already processed upstream, so naturally written complaints differ from training inputs. Long complaints are truncated to the opening 256 tokens. Blank, punctuation-only and over-limit inputs produce a clear message and remove the previous prediction. Avoid submitting private account information.

## Reproducibility

Source: [Consumer Complaints Dataset for NLP, version 1](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp). After quality checks and deduplication, 123,978 records were split into 99,182 train, 12,398 validation and 12,398 test. Training-only vocabulary and class weighting were used; identical encoded inputs do not cross splits.

The tokenizer, preprocessing, labels and selected weights are bundled locally and integrity-checked at startup. The app runs CPU inference and does not retrain or download a dataset/model at startup.

For local use: install the included requirements in Python 3.11, then run `python app.py`.
