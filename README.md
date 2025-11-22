# Interpertable-Machine-Learning-SHAP-Analysis-of-credit-risk-PREDICTION
 SHAP Analysis of credit risk PREDICTION
CREDIT RISK PREDICTION USING INTERPRETABLE MACHINE LEARNING

This project builds a complete credit default risk prediction system using machine learning with a strong focus on interpretability. The goal is to accurately predict borrower default risk while ensuring that predictions remain transparent, explainable, and suitable for financial or regulatory environments.

PROJECT HIGHLIGHTS
- High-performance LightGBM model
- Calibrated probability outputs
- Optimized decision threshold for imbalanced data
- Full set of SHAP and LIME explanations
- Global and local interpretability reports
- ROC, PR curve, and confusion matrix for evaluation
- A detailed analytical report is included

PROJECT FILES
best_model_lgb.joblib
calibrated_model.joblib
chosen_threshold.txt
credit_risk_prediction.ipynb
shap_summary.png
force_plot_*.png
confusion_matrix.png
roc.png
pr.png
local_shap_reports.json
local_lime_reports.json
global_shap_all.csv
report.md

PROBLEM STATEMENT
Financial institutions must identify high-risk borrowers while maintaining fairness and transparency. This project solves the challenge by combining strong predictive modeling with interpretable machine learning techniques.

MACHINE LEARNING APPROACH
1. Data Processing
2. Model Training
3. Evaluation
4. Threshold Optimization

MODEL INTERPRETABILITY
Includes SHAP global and local explanations, LIME insights, and global SHAP CSV analysis.

VISUALIZATIONS
Includes confusion matrix, ROC curve, PR curve.

TECH STACK
Python, LightGBM, Scikit-learn, Pandas, NumPy, SHAP, LIME, Matplotlib.

HOW TO RUN
pip install -r requirements.txt
jupyter notebook credit_risk_prediction.ipynb

RESULT SUMMARY
LightGBM delivered strong predictive accuracy with transparent SHAP and LIME explanations.



