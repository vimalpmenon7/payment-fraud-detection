# payment-fraud-detection

![CI](https://github.com/vimalpmenon7/payment-fraud-detection/actions/workflows/ci.yml/badge.svg)

An end-to-end **machine learning pipeline for card-payment fraud detection** in Python and scikit-learn: leakage-safe feature engineering, chronological train/validation/test evaluation, model comparison, **cost-based threshold selection**, permutation feature importance, and model persistence with a scoring CLI.

## Problem

Fraud is rare (about 1 to 2 percent of transactions here), so accuracy is meaningless and random train/test splits leak the future. A fraud model must be judged on precision-recall behaviour and on what a missed fraud versus a manual review actually costs.

## Solution

```
synthetic transactions --> feature engineering (past-only customer history)
        --> chronological split: train 70% | validation 15% | test 15%
        --> fit 3 models (logistic regression, random forest, gradient boosting)
        --> pick best by validation PR-AUC
        --> choose decision threshold that minimises expected cost on validation
        --> report on the untouched test period + permutation importance
        --> save model.joblib + report.json  -->  score new transactions (CLI)
```

## Key engineering choices

- **No future leakage:** history features (transactions in the last 24h, hours since last transaction, amount versus the customer's previous average) use only earlier rows. A unit test proves that features of early rows are identical with or without later data.
- **Time-based split:** the test set is strictly later than training and validation data.
- **Imbalance handled and measured honestly:** class weights, PR-AUC as the selection metric, and ROC-AUC reported alongside it.
- **Business cost, not 0.5:** the threshold minimises missed-fraud loss (the transaction amount) plus a fixed review cost per flagged transaction.
- **Explainability:** permutation importance on the test period.

## Features

Amount, log amount, hour, night flag, new-device flag, country mismatch, merchant category (one-hot), `txn_count_24h`, `hours_since_last`, `first_txn`, `amount_to_cust_mean`.

## Tech stack

Python 3.10+, pandas, NumPy, scikit-learn, joblib, GitHub Actions, unittest.

## Project structure

```
frauddet/
  data.py       synthetic transaction generator with fraud patterns and bursts
  features.py   leakage-safe feature engineering
  model.py      split, training, model selection, cost-based threshold, save/load/score
  cli.py        train and score commands
tests/          10 unit tests
```

## Usage

```bash
git clone https://github.com/vimalpmenon7/payment-fraud-detection.git
cd payment-fraud-detection
pip install -r requirements.txt
python -m unittest discover -s tests -t .      # 10 tests
python -m frauddet.cli train --out artifacts   # trains, prints the report, saves artifacts/
python -m frauddet.cli score --model artifacts --input my_transactions.csv
```

`score` expects the columns `txn_id, customer_id, ts, amount, merchant_category, device_new, country_mismatch` and prints `txn_id, fraud_probability, flagged`.

## Results (synthetic data, seed 7)

20,000 requested transactions became 16,323 after filtering and bursts, 240 of them fraud (1.5%). The test period contains about 1.2% fraud.

| Model | Validation ROC-AUC | Validation PR-AUC |
|---|---|---|
| Logistic regression (selected) | 0.963 | 0.659 |
| Random forest | 0.924 | 0.581 |
| Hist. gradient boosting | 0.932 | 0.572 |

Test period at the cost-optimal threshold (0.76):

| ROC-AUC | PR-AUC | Precision | Recall | Flagged | Expected cost | Cost with no model |
|---|---|---|---|---|---|---|
| 0.933 | 0.573 | 0.196 | 0.724 | 4.4% | 1,175 | 3,406 |

Top permutation-importance features: `device_new`, `country_mismatch`, `log_amount`, `cat_gift_cards`, `cat_crypto`.

## Limitations (please read)

- **All data is synthetic.** The fraud rules are written by me, so these numbers show the pipeline works and are **not** evidence of real-world fraud-detection performance. Logistic regression wins partly because the generator itself is a logistic risk model.
- The test period contains only about 35 fraud cases, so metrics have wide uncertainty; no confidence intervals or repeated runs are reported.
- No cross-validation, hyperparameter search, probability calibration or drift monitoring.
- Not deployed: there is no API, container, cloud (for example AWS SageMaker) or experiment tracking (for example MLflow) in this repo.
- Costs (review cost 5, missed fraud equals the amount) are illustrative assumptions.

## Future improvements

- Real or public benchmark data (for example the Kaggle credit-card fraud dataset) with repeated time-based evaluation.
- Experiment tracking with MLflow; serve the model behind a FastAPI endpoint in a Docker image; deploy on AWS.
- Calibration, drift monitoring and a feature store for customer history.

## License

MIT
