# Credit Risk Prediction on FICO HELOC Data

This project builds and evaluates credit risk models on the FICO HELOC dataset.  
The goal is to predict whether a borrower is **Good (0)** or **Bad (1)** risk using credit bureau features.

---

## Data

All data lives under `data/`:

- `data/raw/FICO_uncleaned.csv` – original FICO HELOC data.
- `data/raw/heloc_data_dictionary-2.*` – data dictionary from FICO.
- `data/processed/FICO_cleaned.csv` – cleaned full dataset (NaN handling + label encoding).
- `data/processed/FICO_cleaned2.csv` – alternative cleaning (XGBoost handles NaNs).
- `data/processed/FICO_reduced.csv` – reduced feature set based on correlation.
- `data/FICO/FICO.csv` – additional FICO-formatted data (if needed).

Key cleaning steps:

- Replace special missing codes (`-9`, `-8`, `-7`) with `NaN` or a sentinel value.
- Drop rows with missing target; let XGBoost handle remaining NaNs.
- Encode `RiskPerformance` (`Good`/`Bad`) to binary and, for modeling, flip so that:
  - `Bad = 1`, `Good = 0`.

---

## Repository Structure

```text
CMPE257-Project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── FICO/
├── docs/
│   ├── figures/
│   ├── metrics/
│   ├── reports/
│   └── slide_decks/
├── models/                             # Trained models / artifacts (optional)
├── notebooks/
│   ├── 01-eda.ipynb                    # Exploratory data analysis
│   ├── 02-cleaning.ipynb               # Data cleaning & label encoding
│   ├── 03-feature-engineering.ipynb    # Feature selection + monotonic constraints
│   ├── 04-train.ipynb                  # Model training (XGBoost)
│   ├── 05-evaluate.ipynb               # Evaluation, threshold tuning, diagnostics
│   ├── 06-inference-demo.ipynb         # Example inference pipeline
│   └── 90-experiments-template.ipynb
├── requirements.txt
└── README.md
```
---

## Setup

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scriptsctivate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Run notebooks from the project root so relative paths to `data/` resolve correctly.

---

## Workflow

### 1. Data Cleaning (`02-cleaning.ipynb`)

- Load `data/raw/FICO_uncleaned.csv`.
- Replace special codes (`-9`, `-8`, `-7`) with `NaN` (or sentinel in the first approach).
- Drop rows with missing `RiskPerformance`; keep feature-side NaNs for XGBoost.
- Label-encode `RiskPerformance` and save:
  - `FICO_cleaned.csv` (first cleaning approach).
  - `FICO_cleaned2.csv` (preferred, NaNs left for XGBoost).

### 2. Feature Engineering & Constraints (`03-feature-eng.ipynb`)

- Compute correlations and export a reduced feature set:
  - `FICO_reduced.csv` with features strongly correlated to `RiskPerformance`.
- Compute Spearman correlations between each feature and target (Bad = 1).
- Build XGBoost **monotonic constraints**:
  - `+1` if higher feature values should increase risk.
  - `-1` if higher feature values should decrease risk.
  - `0` for weak/no relationship.

### 3. Exploratory Data Analysis (`01-eda.ipynb`)

- Use `pandas` and `ydata-profiling` on raw and cleaned data.
- Inspect:
  - Schema, missingness, and distributions.
  - Class balance of `RiskPerformance`.
  - Key feature distributions and correlations.

### 4. Model Training (`04-train.ipynb`)

- Load `FICO_cleaned2.csv` and prepare:
  - `X` = features, `y` = `RiskPerformance` (flipped: Bad = 1).
- Perform stratified train/validation/test splits.
- Train an `XGBClassifier` with:
  - `objective="binary:logistic"`, `eval_metric="auc"`, `tree_method="hist"`.
  - Early stopping on validation AUC.
  - Monotonic constraints from feature engineering.
  - Class weighting via `scale_pos_weight` for imbalance.

### 5. Evaluation & Threshold Tuning (`05-evaluate.ipynb`)

- Rebuild the same splits and load the best model.
- Evaluate on the test set:
  - ROC curve, ROC AUC.
  - Precision–recall curve, average precision.
  - Confusion matrices.
  - Metrics across thresholds (accuracy, precision, recall, F1).
- Search thresholds to optimize F1 vs. default `0.5`.
- Summarize objectives (e.g., AUC ≥ 0.75, recall/precision targets, baselines) and mark PASS/FAIL.
- Analyze feature importance and misclassifications (false positives/negatives).

Typical performance (XGBoost with tuned parameters and monotonic constraints):

- Accuracy ≈ **0.72**
- ROC AUC ≈ **0.79**  
(values vary slightly with random seed and hyperparameters).

### 6. Inference Demo (`06-inference-demo.ipynb`)

- Load the trained model and required preprocessing.
- Score new observations:
  - Compute probability of `Bad` (1).
  - Apply chosen decision threshold.
- Can be extended into a script, API, or simple UI for batch scoring.

---

## Possible Extensions

- Compare with alternative models (Logistic Regression, LightGBM, CatBoost).
- Probability calibration (e.g., `CalibratedClassifierCV`).
- Cost-sensitive threshold selection based on business loss functions.
- Export a production-ready model and wrap it in a CLI or REST service.
