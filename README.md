## Loan Approval Prediction: Exploratory Data Analysis & Machine Learning

This project involves data exploration and machine learning modeling for loan approval prediction using a tabular dataset. The objective is to analyze the data, identify patterns and relationships, and build a model to predict whether a loan application should be approved.

### Files

* `code_loan(2).ipynb` — Jupyter Notebook containing the entire workflow:

  * Data cleaning
  * Feature selection
  * Preprocessing
  * Model training and evaluation

* `eda_loan.ipynb` — Exploratory Data Analysis notebook (distributions, correlations, class imbalance)
* `loan_data.csv` — The dataset used for analysis (expected to be in the same directory)
* `report.pdf` — Project summary report
* `roc_curves.png` — ROC curve comparison across all models

NOTE: The pickle file storing the model is too large to be pushed to the repository

---

### Features

* **EDA** using pandas, matplotlib, seaborn to understand data distributions, correlations, and class imbalance.
* **Data preprocessing** including one-hot encoding, feature scaling, and target leakage detection.
* **Feature selection** using SelectKBest, RFE, and Lasso.
* **Model training** using machine learning classifiers such as:

  * XGBoost
  * Random Forest
  * Logistic Regression
  * Support Vector Machine (SVM)
  * KNN
* **Model evaluation** using ROC-AUC, precision, recall, F1-score, and confusion matrices.
* **SMOTE** and **class-weighting** used for balancing the imbalanced dataset.
* **Fairness audit** across gender and education groups.

---

### Results

The notebook evaluates the performance of multiple models and compares them based on ROC-AUC, precision, and recall. The final model (XGBoost, tuned) achieves a ROC-AUC of 0.938 and 87.2% recall on high-risk applicants who should not be approved.

---

📌 Author

Ananya
