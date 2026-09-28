# CodeAlpha_CreditScoring

Machine Learning Internship - **Task 1: Credit Scoring Model** (CodeAlpha)

## Objective
Predict an applicant's creditworthiness (good vs bad credit risk) from past financial data.

## Dataset
German Credit Data (UCI ML Repository / OpenML `credit-g`): 1000 applicants, 20 features
(checking status, credit history, duration, credit amount, savings, employment, age, housing, etc.).
Target: `bad` credit = 1 (high risk).

## Approach
1. Load data, check class balance (about 30% bad credit)
2. **Feature engineering**: `credit_per_month`, `credit_to_age`, `is_long_term`, `is_young`, `log_credit_amount`
3. Preprocessing: StandardScaler for numeric, OneHotEncoder for categorical (inside a Pipeline)
4. Stratified 80/20 split, `class_weight="balanced"` to handle imbalance
5. Models: Logistic Regression, Decision Tree, Random Forest
6. Metrics: Precision, Recall, F1-Score, ROC-AUC (plus 5-fold CV ROC-AUC)

## Results (test set)
> Run the script and paste the table from `results.csv` here.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |

## Run
```bash
pip install -r requirements.txt
python credit_scoring.py
```
Needs internet the first time (downloads the dataset from OpenML).
Outputs: `results.csv`, `confusion_matrix.png`, `roc_curves.png`, `feature_importance.png`

## Tech
Python, pandas, scikit-learn, matplotlib, seaborn
