# Machine Learning for Asset Management – Tutorial 3

This repository contains the implementation of Tutorial 3 from the ESILV course *Machine Learning for Asset Management*.

## Project Overview

This project is divided into two main parts:

### Risk Budgeting & Equal Risk Contribution (ERC)

- Implementation of Equal Risk Contribution (ERC) portfolios
- Long-only optimization under volatility constraint
- Long/short optimization
- Market-neutral long/short optimization
- Rolling ERC portfolios using historical covariance matrices
- Numerical optimization with `scipy.optimize.minimize`

### Random Forest Modeling

- Data cleaning and missing value imputation (IterativeImputer)
- Random Forest regression for forward return prediction
- Feature importance analysis
- Proximity matrix computation
- Individual tree prediction visualization
- Hyperparameter analysis:
  - Number of trees
  - Number of splitting variables (mtry / max_features)
- Out-of-bag error evaluation

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib & Seaborn

## Key Concepts Covered

- Risk contribution and risk parity
- Log-barrier optimization
- Random Forest regression
- Model tuning and cross-validation
- Portfolio construction under constraints

## Academic Context

ESILV – Financial Engineering  
Course: Machine Learning for Asset Management  
Level: Master 1
