# House-SalePrice-Prediction-Based-on-Lasso-and-Random-Forest

# Overview

Predict house sale prices using Lasso Regression and Random Forest Regression. This project demonstrates data preprocessing, model training, hyperparameter tuning, and prediction on new examples.

# Dataset

The dataset is stored in an Excel file `price.xlsx`.

Numerical Features: `MSSubClass`, `LotArea`, `OverallCond`, `YearRemodAdd`, `BsmtFinSF2`, `TotalBsmtSF`

Target: `SalePrice`

# Data Preprocessing

Feature selection – select relevant numerical and categorical features.

Missing value handling – remove rows with NaNs in selected features.

Encoding categorical features – one-hot encoding for categorical variables (MSZoning).

Train-test split – 80% of data for training, 20% for testing.

# Models

1. Lasso Regression

Linear model with L1 regularization.

Hyperparameter tuning with `GridSearchCV` for `alpha` in `[0.001, 0.01, 0.1, 1, 10]`.

Outputs best `alpha` and R² scores on training and test sets.

2. Random Forest Regression

Ensemble model using 100 decision trees.

Captures non-linear relationships and feature interactions.

Provides training and test R² scores.

# License

This project is licensed under the MIT License.
