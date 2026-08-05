# Fraudulent Bank Account Detection 

Binary classification on the **Bank Account Fraud (BAF)** "Base" dataset
(1,000,000 applications, target `fraud_bool`, 1.10% fraud, ~90:1 imbalance).

## Deliverables

| File | What it is |
|------|------------|
| `Fraud_Detection_Paper.pdf` | 5-page IEEE-style two-column conference paper (compiled) |
| `Fraud_Detection_Slides.pptx` / `.pdf` | 9-slide, ~10-minute presentation |
| `code/` | All Python scripts + LaTeX source (`paper.tex` + `figures/`) |
| `tables/` | Comparison table, 6 ablation tables, preprocessing summary (CSV) |
| `figures/` | All EDA + evaluation figures (PNG) |
| `models/` | Serialized best estimators + preprocessor + run metadata (JSON) |

## Pipeline (run order)

1. `code/eda.py`      — full 1M-row EDA; summary JSON + Phase-1 figures
2. `code/prep.py`     — preprocessing; caches transformed arrays + `preprocessor.pkl`
3. `code/fit.py MODEL`— per-model grid-search CV (PR-AUC); MODEL ∈
   {LogReg, DecisionTree, KNN, SVM, RandomForest, HistGB, Stacking}
4. `code/evaluate.py` — held-out metrics, comparison table, ROC/PR/confusion figures
5. `code/build_slides.js` — generates the deck (needs `pptxgenjs`)

## Reproducibility

- **Seed 42** for sampling, splitting, CV folds, and every stochastic model.
- 50,000-row stratified subsample (fraud rate preserved) for SVM/KNN tractability;
  80/20 stratified split → train 40,000 (441 fraud) / test 10,000 (110 fraud).
- Python 3.12, scikit-learn 1.8.0, NumPy, pandas, matplotlib.
- `HistGradientBoostingClassifier` stands in for LightGBM/XGBoost (offline equivalent).
- SMOTE implemented from scratch (fixed RNG), applied to KNN training data only.

## Key results (held-out test set, sorted by PR-AUC)

| Model | ROC-AUC | PR-AUC | Recall@5%FPR | Accuracy |
|-------|--------:|-------:|-------------:|---------:|
| Logistic Regression       | 0.909 | 0.136 | 0.536 | 0.803 |
| Histogram Grad. Boosting  | 0.889 | 0.136 | 0.491 | 0.909 |
| Stacking ensemble         | 0.913 | 0.135 | 0.527 | 0.818 |
| Random Forest             | 0.897 | 0.127 | 0.500 | 0.983 |
| SVM (RBF)                 | 0.896 | 0.097 | 0.500 | 0.832 |
| Decision Tree             | 0.781 | 0.060 | 0.291 | 0.777 |
| k-NN (SMOTE)              | 0.800 | 0.046 | 0.291 | 0.660 |

Headline findings: (1) a weighted Logistic Regression matches gradient boosting and
stacking because the anonymized features behave additively with weak interactions;
(2) the **missing-previous-address indicator** is the single strongest predictor
(standardized weight +1.33); (3) k-NN's CV PR-AUC of 0.988 collapsing to 0.046 on
real data is a textbook **SMOTE-before-CV leakage** artifact; (4) Random Forest's 98%
accuracy is the imbalance "accuracy mirage" — it catches only 20% of fraud at a 0.5
threshold, which is why PR-AUC and budgeted recall are the right metrics.

## Limitations

50k subsample (441 training fraud) caps achievable PR-AUC; single train/test split;
SVM tuned/refit on subsamples; no probability calibration or fairness audit yet.
