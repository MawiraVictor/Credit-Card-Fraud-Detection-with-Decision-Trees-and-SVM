# Credit Card Fraud Detection with Decision Trees and SVM

A machine learning project that builds and compares two classification models, a Decision Tree and a Support Vector Machine, to detect fraudulent credit card transactions. The work is contained in the Jupyter notebook `decision_tree_svm_ccFraud.ipynb`.

## Overview

Credit card fraud detection is a binary classification problem in which transactions must be labeled as legitimate (class 0) or fraudulent (class 1). The dataset is highly imbalanced: only 492 of 284,807 transactions are fraudulent, representing approximately 0.172 percent of all records. This imbalance requires special handling during training and evaluation.

The notebook demonstrates the full workflow: data loading, exploratory analysis, preprocessing, model training, and evaluation using the ROC-AUC metric.

## Dataset

The project uses the public Credit Card Fraud Detection dataset originally published on Kaggle.

- Source URL used in the notebook: `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv`
- Rows: 284,807
- Columns: 31
- Features: `Time`, `V1` through `V28`, and `Amount`
- Target: `Class` (0 for legitimate, 1 for fraudulent)

Most feature names are anonymized because the original values are the result of a PCA transformation. The `Time` column is excluded from modeling.

## Requirements

The notebook installs and uses the following Python packages:

- pandas 2.2.3
- scikit-learn 1.6.0
- matplotlib 3.9.3

A Python 3 environment with Jupyter is required to run the notebook.

## Project Structure

```
Credit Card Fraud Detection with Decision Trees and SVM/
├── decision_tree_svm_ccFraud.ipynb   Main analysis notebook
├── creditcard.csv                    Dataset (not tracked in Git)
├── LICENSE                           License file
└── README.md                         This file
```

## Methodology

### Preprocessing

1. Standard scaling is applied to features 2 through 30 to remove the mean and scale to unit variance.
2. Features are normalized using the L1 norm to support faster model convergence.
3. The `Time` column is excluded.
4. The data is split into training and test sets with a 70/30 ratio and a fixed random state for reproducibility.

### Class Imbalance Handling

- Decision Tree: balanced sample weights are computed with `compute_sample_weight` and passed to the `fit` method.
- SVM: the `class_weight='balanced'` parameter is used directly.

### Models

**Decision Tree Classifier**
- Maximum depth: 4
- Random state: 35
- Trained with balanced sample weights

**Linear Support Vector Machine**
- Loss: hinge
- `fit_intercept`: False
- `class_weight`: balanced
- Random state: 31

### Evaluation Metric

The ROC-AUC score is used to measure each model's ability to distinguish between fraudulent and legitimate transactions across all probability thresholds. It is well suited to imbalanced datasets.

## Results

| Model | ROC-AUC Score |
|-------|---------------|
| Decision Tree | 0.939 |
| SVM | 0.986 |

The SVM achieved the higher ROC-AUC score when all 30 features were used.

## Practice Exercises

The notebook includes five exercises that explore feature selection and its effect on each model.

1. Identify the top six features by absolute correlation with the target class. The result is V3, V10, V12, V14, V16, and V17.
2. Modify the input feature matrix to use only these six features.
3. Retrain the Decision Tree and observe the change in ROC-AUC.
4. Retrain the SVM and observe the change in ROC-AUC.
5. Draw conclusions about the two model families.

Key observations from the exercises:

- With the full feature set, the SVM performed better than the Decision Tree.
- The Decision Tree benefited from feature selection and improved its score.
- The SVM required higher feature dimensionality to construct an effective decision hyperplane and its score decreased with fewer features.

## Reproducing the Results

1. Clone the repository.
2. Ensure Python 3 and Jupyter are installed.
3. Install the required packages, either through the first cell of the notebook or manually with pip.
4. Place `creditcard.csv` in the project root. The notebook can also download the dataset directly from the URL shown above.
5. Open `decision_tree_svm_ccFraud.ipynb` and run all cells in order.

Note that the dataset is not tracked in the Git repository because it exceeds GitHub's file size limits. Download it separately before running the notebook locally.

## License

See the `LICENSE` file in the repository for licensing information.

## Authors

- Abhishek Gagneja
- Jeff Grossman (contributor)

This project is based on course material from IBM Skills Network.
