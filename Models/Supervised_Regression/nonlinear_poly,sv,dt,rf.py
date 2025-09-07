import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

def prepare_data(features_path, labels_path, test_size, random_state,
                 use_scaler=True, poly_degree=None):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()
                   
    if poly_degree is not None and poly_degree > 1:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X = poly.fit_transform(X)
    if use_scaler:
        X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# please replace your grid parameters here.
def get_model_and_params(method, random_state=42):
    if method == 'poly':
        return LinearRegression(), None
    elif method == 'svr':
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
        return SVR(), param_grid
    elif method == 'dt':
        param_grid = {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        return DecisionTreeRegressor(random_state=random_state), param_grid
    elif method == 'rf':
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
        return RandomForestRegressor(random_state=random_state), param_grid
    else:
        raise ValueError("Unsupported method")


def train_and_evaluate(method, X_train, y_train, X_test, y_test, search='grid', cv=5, random_state=42, poly_degree=None):
    if method == 'poly':
        model = LinearRegression()
        start = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start
        best_params = None
    else:
        model, param_grid = get_model_and_params(method, random_state)
        if param_grid is None:
            start = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start
            best_params = None
        else:
            start = time.time()
            grid_search = GridSearchCV(model, param_grid, cv=cv)
            grid_search.fit(X_train, y_train)
            duration = time.time() - start
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_

    y_pred = model.predict(X_test)
    results = {
        'R2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        'MAE': mean_absolute_error(y_test, y_pred),
        'train_time_sec': duration,
        'best_params': best_params
    }
    return results

if __name__ == "__main__":
    features_path = 'features.csv' 
    labels_path = 'labels.csv'     
    test_size = 0.2
    random_state = 42
    use_scaler = True
    poly_degree = 2

    method = 'svr'  # Choose a model: 'poly', 'svr', 'dt', 'rf'

    reports = {}
        
    # Polynomial regression requires polynomial feature transformation
    degree = poly_degree if method == 'poly' else None
    X_train, X_test, y_train, y_test = prepare_data(features_path, labels_path, test_size, random_state, use_scaler, poly_degree=degree)
    results = train_and_evaluate(method, X_train, y_train, X_test, y_test, poly_degree=degree)

    print("\nSummary Report:")
    print(f"Model: {method}")
    print(f" Training time (s): {res['train_time_sec']:.3f}")
    print(f" Best parameters: {res['best_params']}")
    print(f" R2: {res['R2']:.4f}")
    print(f" MSE: {res['MSE']:.4f}")
    print(f" RMSE: {res['RMSE']:.4f}")
    print(f" MAE: {res['MAE']:.4f}")
    print("-")
