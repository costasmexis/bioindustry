import pandas as pd
import numpy as np
import glob
import os
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

N_SPLITS = 10
FOLDER = 'results'  

# Imort data
def import_data(path):
    df = pd.read_excel(path, header=[0,1])
    df = df[[('Temperature', 'norm'), ('mu', 'norm'), ('qp', 'norm')]]
    print(f'Dataset shape: {df.shape}')

    # Split data to X and y
    X = df[[('Temperature', 'norm'), ('mu', 'norm')]]
    y = df[[('qp', 'norm')]]

    return df, X, y

# Function to manually perform cross-validation
def manual_cv(model, model_name: str, X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=False)
    
    for iter, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_index].values, X.iloc[val_index].values
        y_train, y_val = y.iloc[train_index].values.ravel(), y.iloc[val_index].values.ravel()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Calculate MSE
        mse = mean_squared_error(y_val, y_pred)

        # DataFrame with y_val and y_pred for each iteration
        df = pd.DataFrame({'y_val': y_val, 'y_pred': y_pred})
        df['iter'] = iter+1
        df['model'] = model_name
        df['sample_id'] = val_index
        df['mse'] = mse

        df.to_csv(f'results/predictions_{iter+1}_{model_name}.csv', index=False)

# Function to read and concat all csv from a folder
def read_csv(folder):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    return pd.concat(df_from_each_file, ignore_index=True)

# Function that tunes a model and return best estimator
def tune_model(model, X, y, params, cv=10, scoring='neg_mean_squared_error'):
    grid = GridSearchCV(model, params, cv=cv, scoring=scoring)
    grid.fit(X, y)
    return grid.best_estimator_

# Create a function for SVR model
def svr_model(X, y):
    X = X.values
    y = y.values.ravel()

    model = SVR()
    
    # Define a grid of parameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1, 10, 100],
        'kernel': ['rbf', 'sigmoid', 'linear']
    }

    best_model = tune_model(model, X, y, param_grid)
    return best_model

# Create a function for Random Forest model
def rf_model(X, y):
    X = X.values
    y = y.values.ravel()

    model = RandomForestRegressor()

    # Define a grid of parameters
    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 10]
    }

    best_model = tune_model(model, X, y, param_grid)
    return best_model

# Create a function for Linear Regression model
def lr_model(X, y):
    X = X.values
    y = y.values.ravel()

    model = LinearRegression()
    model.fit(X,y)
    return model

