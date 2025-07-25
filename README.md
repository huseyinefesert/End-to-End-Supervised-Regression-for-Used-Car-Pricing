# End-to-End-Supervised-Regression-for-Used-Car-Pricing

This project is an end-to-end supervised machine learning pipeline for predicting used car prices based on a diverse set of features, using various regression algorithms and real-world data.

## 🚗 Dataset

- **cars_en.csv**: Contains detailed records for used cars, including brand, model, year, mileage, engine details, color, body type, fuel type, and more.
- The dataset was preprocessed to handle missing values, categorical encoding, feature selection, and outlier removal.

## 🛠️ Workflow Overview

1. **Data Cleaning & Feature Engineering**
   - Dropped irrelevant columns (IDs, titles, location, etc.).
   - Filled missing values for categorical columns with the mode.
   - Converted price, year, mileage, engine size, and engine power columns to numeric types and handled special cases.
   - Encoded categorical features with Label Encoding, One Hot Encoding, and mapping where appropriate.
   - Outliers in the target column (Price) were removed using the IQR method.

2. **Exploratory Data Analysis (EDA)**
   - Distributions of key features visualized with matplotlib and seaborn.

3. **Model Building & Evaluation**
   - Multiple regression models compared:  
     - Linear Regression  
     - Lasso & Ridge Regression  
     - KNeighbors Regressor  
     - Decision Tree & Random Forest Regressor  
     - AdaBoost, GradientBoosting, XGBoost, LightGBM  
   - Performance measured by MAE, RMSE, and R² metrics on both training and test sets.

4. **Hyperparameter Tuning**
   - XGBoost hyperparameters optimized using RandomizedSearchCV.

## 📈 Results

| Model                   | Train RMSE | Train MAE | Train R² | Test RMSE | Test MAE | Test R² |
|-------------------------|------------|-----------|----------|-----------|----------|---------|
| Linear Regression       | 147,525    | 108,988   | 0.798    | 147,650   | 110,099  | 0.807   |
| Lasso                   | 147,525    | 108,988   | 0.798    | 147,646   | 110,095  | 0.807   |
| Ridge                   | 147,532    | 109,001   | 0.798    | 147,621   | 110,045  | 0.807   |
| K Neighbors Regressor   | 206,737    | 149,139   | 0.603    | 271,485   | 195,703  | 0.346   |
| Decision Tree           | 2,751      |    138    | 0.9999   | 110,995   |  79,806  | 0.891   |
| Random Forest Regressor | 34,713     |  23,554   | 0.989    |  80,110   |  57,630  | 0.943   |
| AdaBoost Regressor      | 145,950    | 117,764   | 0.802    | 143,607   | 116,444  | 0.817   |
| Gradient Boosting       | 90,154     |  65,159   | 0.925    |  88,352   |  64,496  | 0.931   |
| XGBoost Regressor       | 41,831     |  30,577   | 0.984    |  79,055   |  57,090  | 0.945   |
| LightGBM Regressor      | 64,586     |  46,790   | 0.961    |  76,506   |  55,206  | 0.948   |


## 📦 Key Python Libraries

- `pandas`, `numpy` – data processing
- `scikit-learn` – model training, preprocessing, metrics, hyperparameter search
- `xgboost`, `lightgbm` – advanced regression models
- `matplotlib`, `seaborn` – visualization

## 🏁 How to Run

1. Place `cars_en.csv` in the project directory.
2. Install requirements (see below).
3. Run the notebook or script:
    ```bash
    python supervised_learning.py
    ```

## 💾 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- lightgbm

Install dependencies via:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
