**Name:** NAVEEN US

**Roll Number:** DA25M020

**Course:** DA5401 - Data Analytics Laboratory

## Notebook Overview

This notebook implements and compares several strategies for handling missing data in the UCI Credit Card dataset. The goal is to evaluate how different imputation methods (simple and regression-based) affect the performance of a downstream classifier (logistic regression). The pipeline includes: introducing Missing At Random (MAR) values, applying three imputation strategies, training classifiers, and comparing model performance.

### Primary Objectives
- Create controlled missingness (MAR) in selected columns.
- Apply Median imputation (baseline), Linear Regression imputation, and KNN Regression imputation.
- Train a logistic regression classifier on each imputed dataset and on a listwise-deleted dataset.
- Compare classification metrics to assess the impact of imputation methods.

## Key Components Implemented

Part A: Data Preprocessing & Imputation
- Load `UCI_Credit_Card.csv` and basic inspection.
- Introduce 10% MAR missingness to AGE, BILL_AMT1, PAY_AMT1.
- Dataset A: Median imputation for missing columns (baseline).
- Dataset B: Linear regression to impute AGE, median for other missing columns.
- Dataset C: KNN regression to impute AGE, median for other missing columns.
- Dataset D: Listwise deletion (drop rows with any NaN).

Part B: Model Training & Evaluation
- Standardize features with StandardScaler.
- Train logistic regression on each dataset (A, B, C, D).
- Evaluate using classification_report (precision, recall, f1-score) and accuracy.
- Aggregate results into a summary table for comparison.

Part C: Comparative Analysis
- Tabulate Accuracy, Precision, Recall, and F1-score for the 'default' class.
- Discuss trade-offs between listwise deletion and imputation.
- Discuss whether regression-based imputation provides benefit over simple median imputation.

## Key Findings & Conclusions (summary)
- Imputation methods retained data and produced comparable or slightly better results vs listwise deletion.
- Depending on feature importance, imputed AGE may have little impact on model performance.
- Simple median imputation can be a strong baseline; regression-based methods may not always improve classifier metrics.

## Files & Outputs Saved
- `Assignment 6.ipynb` — main notebook
- `initial_distributions.png` — original distributions plot
- `imputation_comparison.png` — AGE distributions after imputation
- `imputation_comparison_bill_amt_zoomed.png` — BILL_AMT1 distributions (zoomed)
- `README.md` — this file

## How to Run

1. Place `UCI_Credit_Card.csv` in the same directory as the notebook:
   /Users/naveenus/Documents/MTech/Semester 1/Data Analytics Laboratory/Assignments/Assignment 6/
2. Open `Assignment 6.ipynb` in Jupyter Notebook / JupyterLab.
3. Run all cells sequentially to reproduce the preprocessing, imputation, training, evaluation, and plots.

## Requirements

Ensure the Python environment includes:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install with:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Reproducibility Notes
- The notebook uses a fixed random_state (42) for train/test splitting and model initialization where applicable to aid reproducibility.
- Artificial missingness is introduced using numpy.random; run cells multiple times will produce different missing indices unless a random seed is set prior to missingness introduction.

## Contact
For questions about the implementation, refer to the notebook comments and cell outputs.

"""
with open("/Users/naveenus/Documents/MTech/Semester 1/Data Analytics Laboratory/Assignments/Assignment 6/README.md", "w") as f:
    f.write(content)
print("README.md written to project folder.")
