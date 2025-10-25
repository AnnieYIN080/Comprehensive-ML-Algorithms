import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

def prepare_data(features_path, labels_path, test_size, random_state,
                 use_scaler=True, use_pca=False, pca_n_comp=None, use_select=False, k_best=None, poly_degree=None):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()
    
    if poly_degree is not None and poly_degree > 1:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X = poly.fit_transform(X)
    if use_scaler:
        X = StandardScaler().fit_transform(X)
    if use_pca and pca_n_comp is not None:
        X = PCA(n_components=pca_n_comp).fit_transform(X)
    if use_select and k_best is not None:
        X = SelectKBest(f_regression, k=k_best).fit_transform(X, y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Please change your parameters below
def build_model(method, alpha=1.0, l1_ratio=0.5, random_state=42):
    param_grid = {}
      
    if method == 'lr':
        return LinearRegression()
    elif method == 'ridge':
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
        return Ridge(alpha=alpha, random_state=random_state), param_grid
    elif method == 'lasso':
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
        return Lasso(alpha=alpha, random_state=random_state), param_grid
    elif method == 'elastic':
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.9]}
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state), param_grid
    elif method == 'poly':
        # Polynomial regression actually involves preprocessing features and then fitting them using linear regression
        return LinearRegression(), None
    elif method == 'svr':
        param_grid = {
                'kernel': ['linear', 'rbf', 'poly'],
                'C': [0.1, 1, 10, 100],
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
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        return RandomForestRegressor(random_state=random_state), param_grid 
    else:
        raise ValueError("Unsupported method")



def train_and_tune_model(method, X_train, y_train, search='grid', cv=5, random_state=42, n_iter=10):
    model, param_grid = build_model(method, random_state=random_state)
    start = time.time()
    
    if method in ['lr', 'poly'] or param_grid is None:
        model.fit(X_train, y_train)
        duration = time.time() - start
        best_params = None
    
    else:
        if search == 'grid':
            searcher = GridSearchCV(model, param_grid, cv=cv)
        else:
            searcher = RandomizedSearchCV(model, param_grid, cv=cv, random_state=random_state, n_iter=n_iter)
        
        searcher.fit(X_train, y_train)
        duration = time.time() - start
        model = searcher.best_estimator_
        best_params = searcher.best_params_
      
    return model, duration, best_params


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'R2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        'MAE': mean_absolute_error(y_test, y_pred),
    }


if __name__ == "__main__":
    features_path = 'features.csv'
    labels_path = 'labels.csv'
    test_size = 0.2
    random_state = 42
    use_scaler = True
    use_pca = False
    pca_n_comp = None
    use_select = False
    k_best = None
    poly_degree = 2
    methods = ['lr', 'ridge', 'lasso', 'elastic', 'poly', 'svr', 'dt', 'rf']
    # if you only want to run lr, can use the command: methods = ['lr']
    search = 'grid'     # 'grid' or 'random'
    cv = 5
    n_iter = 10
  
    reports = {}
    for method in methods:
        print(f"Training and evaluating model: {method}")
        degree = poly_degree if method == 'poly' else None
        X_train, X_test, y_train, y_test = prepare_data(features_path, labels_path, test_size, random_state,
                                                       use_scaler, use_pca, pca_n_comp, use_select, k_best, poly_degree=degree)
        model, train_time, best_params = train_and_tune_model(method, X_train, y_train, search=search, cv=cv, random_state=random_state, n_iter=n_iter)
        metrics = evaluate_model(model, X_test, y_test)
        reports[method] = {
            'train_time_sec': train_time,
            'best_params': best_params,
            **metrics
            # Through metrics, unpack all key-value pairs in this dictionary and merge them into the reports[method] dictionary
        }

    print("\nSummary Report:")
    for method, res in reports.items():
        print(f"Model: {method}")
        print(f" Training time (s): {res['train_time_sec']:.3f}")
        print(f" Best parameters: {res['best_params']}")
        print(f" R2: {res['R2']:.4f}")
        print(f" MSE: {res['MSE']:.4f}")
        print(f" RMSE: {res['RMSE']:.4f}")
        print(f" MAE: {res['MAE']:.4f}")
        print("-")
