import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import warnings
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

df = pd.read_csv("cars_en.csv")

df.drop("ListingID", axis=1, inplace=True)
df.drop("ListingTitle", axis=1, inplace=True)
df.drop("ListingDate", axis=1, inplace=True)
df.drop("District", axis=1, inplace=True)
df.drop("City", axis=1, inplace=True)
df.drop("TradeInAvailable", axis=1, inplace=True)
df.drop("SellerType", axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df.drop(6526, inplace=True)
df.drop(3654, inplace=True)
df.drop(657, inplace=True)

most_common = df["BodyType"].mode()[0]
df["BodyType"] = df["BodyType"].fillna(most_common)
most_common = df["Color"].mode()[0]
df["Color"] = df["Color"].fillna(most_common)
df["Price(TRY)"] = (
    df["Price(TRY)"]
    .astype(str)                
    .str.replace("[^0-9]", "", regex=True)  
    .astype(int)                
)

df["Year"] = (
    df["Year"]
    .astype(str)                
    .str.replace("[^0-9]", "", regex=True)  
    .astype(int)                
)

df["Mileage(km)"] = (
    df["Mileage(km)"]
    .astype(str)                
    .str.replace("[^0-9]", "", regex=True)  
    .astype(int)                
)

def clean_engine_size(value):
    if pd.isnull(value):
        return np.nan
    value = str(value).lower()
    
    if "kadar" in value or "altı" in value:
        return 1000
    
    if '-' in value:
        first_num = value.split('-')[0]
        num = ''.join(filter(str.isdigit, first_num))
        return int(num) if num else np.nan
    
    num = ''.join(filter(str.isdigit, value))
    return int(num) if num else np.nan


df['EngineSize(cc)'] = df['EngineSize(cc)'].apply(clean_engine_size)
median_value = df["EngineSize(cc)"].median()
df["EngineSize(cc)"] = df["EngineSize(cc)"].fillna(median_value)
df['EngineSize(cc)'] = df['EngineSize(cc)'].astype(int)

def clean_engine_power(value):
    if pd.isnull(value):
        return np.nan
    value = str(value).lower()
    
    if "kadar" in value or "altı" in value:
        return 50
    
    if '-' in value:
        first_num = value.split('-')[0]
        num = ''.join(filter(str.isdigit, first_num))
        return int(num) if num else np.nan
    
    num = ''.join(filter(str.isdigit, value))
    return int(num) if num else np.nan

df['EnginePower(HP)'] = df['EnginePower(HP)'].apply(clean_engine_power)
median_value = df["EnginePower(HP)"].median()
df["EnginePower(HP)"] = df["EnginePower(HP)"].fillna(median_value)
df['EnginePower(HP)'] = df['EnginePower(HP)'].astype(int)

most_common = df["DriveTrain"].mode()[0]
df["DriveTrain"] = df["DriveTrain"].fillna(most_common)

df = df.drop("VehicleTax(TRY)", axis=1)
df = df.drop("AccidentHistory", axis=1)

le = LabelEncoder()
df['Brand'] = le.fit_transform(df['Brand'])
df['Brand'] = df['Brand'].astype(int)

df.drop("Model", axis=1, inplace=True)
df.drop("PaintAndPartsCondition", axis=1, inplace=True)
df['Series'] = le.fit_transform(df['Series'])
df['Series'] = df['Series'].astype(int)

brand_map = {'Manual': 0, 'Semi-Automatic': 1, 'Automatic': 2}
df['TransmissionType'] = df['TransmissionType'].map(brand_map)

df = pd.get_dummies(df, columns=['FuelType'], prefix='FuelType')
df = pd.get_dummies(df, columns=['BodyType'], prefix='BodyType')
df = pd.get_dummies(df, columns=['DriveTrain'], prefix='DriveTrain')

df['Color'] = le.fit_transform(df['Color'])
df['Color'] = df['Color'].astype(int)

fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize=(15,12))
fig.suptitle("Distributions", fontsize = 18, fontweight = "bold")

columns = df.select_dtypes(include=['int', 'float']).columns
for i, col in enumerate(columns):
    row = i // 3
    col_idx = i % 3
    ax = axes[row, col_idx]
    sns.histplot(data = df, x = col, kde=True, ax=ax, bins=30)
    ax.set_title(col, fontsize=10, fontstyle = "italic")

plt.tight_layout()
plt.show() 

def remove_outliers_from_column(df,target_col, threshold = 1.5):
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[ (df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]

#print("original data shape: ", df.shape)
df_target_clean = remove_outliers_from_column(df, "Price(TRY)")
#print("only target column cleaning shape: ", df_target_clean.shape)

X = df_target_clean.drop("Price(TRY)", axis=1)
y = df_target_clean["Price(TRY)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

models = {
    "Linear Regression" : LinearRegression(),
    "Lasso" : Lasso(),
    "Ridge" : Ridge(),
    "K Neighbors Regressor" : KNeighborsRegressor(),
    "Decision Tree" : DecisionTreeRegressor(),
    "Random Forest Regressor" : RandomForestRegressor(),
    "Adaboost Regressor" : AdaBoostRegressor(),
    "Gradient Boost Regressor" : GradientBoostingRegressor(),
    "XGBoost Regressor" : XGBRegressor(),
    "LightGBM Regressor" : LGBMRegressor()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])
    print("Model performance for Training Set")
    print("Root Mean Squared Error: ", model_train_rmse)
    print("Mean Absolute Error: ", model_train_mae)
    print("R2 Score: ", model_train_r2)

    print("-----------------------------------")
    
    print("Model performance for Test Set")
    print("Root Mean Squared Error: ", model_test_rmse)
    print("Mean Absolute Error: ", model_test_mae)
    print("R2 Score: ", model_test_r2)

    print("-----------------------------------")
    print("\n")


xgboost_params = {
        "learning_rate" : [0.1, 0.01],
        "max_depth" : [5,8,12,20,30],
        "n_estimators" : [100,200,300,500],
        "colsample_bytree" : [0.3, 0.4, 0.5, 0.7, 1]
}
randomized_cv = RandomizedSearchCV(estimator=XGBRegressor(), param_distributions=xgboost_params, cv = 5, n_jobs = -1)

randomized_cv.fit(X_train, y_train)
print(randomized_cv.best_params_)

model = XGBRegressor(n_estimators = 300, max_depth = 5, learning_rate = 0.1, colsample_bytree = 0.3)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)
print("Model performance for Training Set")
print("Root Mean Squared Error: ", model_train_rmse)
print("Mean Absolute Error: ", model_train_mae)
print("R2 Score: ", model_train_r2)
print("-----------------------------------")

print("Model performance for Test Set")
print("Root Mean Squared Error: ", model_test_rmse)
print("Mean Absolute Error: ", model_test_mae)
print("R2 Score: ", model_test_r2)
print("-----------------------------------")
print("\n")