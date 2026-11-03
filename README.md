# Credit Card Fraud Detection with Decision Trees and SVM

A machine learning project that compares two classification models, a Decision Tree and a Support Vector Machine, to detect fraudulent credit card transactions.

This repository contains a Jupyter notebook-based analysis and a full workflow for training and evaluating both models on a highly imbalanced credit-card fraud dataset.

## Overview

Credit card fraud detection is a binary classification problem where each transaction is labelled as either legitimate (`0`) or fraudulent (`1`). The dataset is highly imbalanced, with fraud cases representing only a small fraction of all transactions.

The notebook demonstrates the complete data science workflow:

- loading the dataset
- exploratory data analysis
- preprocessing and feature scaling
- handling class imbalance
- training Decision Tree and SVM models
- evaluating model performance with ROC-AUC

## Dataset

The project uses the public Credit Card Fraud Detection dataset from Kaggle/IBM Skills Network.

- Source: `creditcard.csv`
- Total rows: 284,807
- Total columns: 31
- Target column: `Class`
- Fraudulent transactions: 492
- Non-fraudulent transactions: 284,315

Most feature columns are anonymized PCA-transformed values, and the `Time` column is excluded from modeling.
<img width="684" height="305" alt="image" src="https://github.com/user-attachments/assets/0966bea9-6c94-4619-ab89-24dc325e705c" />

Note: the dataset file is intentionally not committed to GitHub because it exceeds GitHub's size limits. Download it separately before running the notebook.

## Requirements

The notebook uses the following Python packages:

- pandas
- scikit-learn
- matplotlib
- Jupyter Notebook / JupyterLab

A Python 3 environment is required.

## Repository Structure

```text
Credit-Card-Fraud-Detection-with-Decision-Trees-and-SVM/
├── Credit Card Fraud Detection with Decision Trees and SVM.ipynb
├── LICENSE
├── README.md
└── creditcard.csv   # download separately
```

## Modeling Approach

### Decision Tree

- Class imbalance handled with `compute_sample_weight`
- Max depth: 4
- Random state: 35

### Support Vector Machine

- Linear SVM with class balancing
- `class_weight='balanced'`
- Random state: 31

## Evaluation Metric

The project uses ROC-AUC, which is well-suited to imbalanced classification problems because it measures performance across all possible decision thresholds.

## Results

| Model | ROC-AUC |
|-------|---------|
| Decision Tree | 0.939 |
| SVM | 0.986 |

The SVM achieved the better score when using the full feature set.

## How to Run

1. Clone the repository.
2. Create a Python environment and install dependencies.
3. Download `creditcard.csv` into the project root.
4. Open the notebook in Jupyter.
5. Run all cells in order.

## License

This project is distributed under the MIT license. See the `LICENSE` file for details.

## Authors

- Abhishek Gagneja
- Jeff Grossman (contributor)

This project is based on IBM Skills Network course material.
