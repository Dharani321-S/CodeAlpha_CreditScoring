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

## Results (test set, positive class = bad credit)

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7656 | 0.745 | 0.5506 | 0.8167 | 0.6577 | 0.8126 |
| Decision Tree | 0.6981 | 0.605 | 0.4021 | 0.6500 | 0.4968 | 0.6672 |
| Random Forest | 0.7987 | 0.790 | 0.6552 | 0.6333 | 0.6441 | 0.8146 |

**Best model by ROC-AUC: Random Forest** (0.8146, accuracy 79%).

## Key Insights
- Random Forest gives the best overall balance (highest accuracy, precision and ROC-AUC).
- Logistic Regression has the highest **recall (81.7%)**: it catches the most risky applicants,
  which matters when missing a defaulter is costly. The trade-off is lower precision (more false alarms).
- A single Decision Tree performs worst and overfits; ensembles generalise better.
- Accuracy alone is misleading here (30% bad credit), so Precision, Recall, F1 and ROC-AUC are reported.

## Run
```bash
pip install -r requirements.txt
python credit_scoring.py
```
Needs internet the first time (downloads the dataset from OpenML).
Outputs: `results.csv`, `confusion_matrix.png`, `roc_curves.png`, `feature_importance.png`

## Tech
Python, pandas, scikit-learn, matplotlib, seaborn
