import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    GradientBoostingRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    import lightgbm as lgb
    LGBMRegressor = lgb.LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor, Pool
    # Pool is a dedicated data structure designed by CatBoost for accelerating and optimizing training. 
    # It is particularly convenient for managing additional information such as category features and sample weights. 
    # It is recommended to use it as the training data input rather than directly using numpy arrays or Dataframes.
except ImportError:
    CatBoostRegressor = None
  

def prepare_data(features_path, labels_path, test_size, random_state, use_scaler=True):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()
    if use_scaler:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_model_and_params(method, random_state=42):
    param_grid = {}
  
    if method == 'gbdt':
        if GradientBoostingRegressor is None:
            raise ImportError("sklearn GradientBoostingRegressor not installed")
        model = GradientBoostingRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
      
    elif method == 'xgboost':
        if XGBRegressor is None:
            raise ImportError("xgboost not installed")
        model = XGBRegressor(random_state=random_state, objective='reg:squarederror')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
      
    elif method == 'lightgbm':
        if LGBMRegressor is None:
            raise ImportError("lightgbm not installed")
        model = LGBMRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [-1, 5, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'feature_fraction': [0.8, 1.0]
        }
      
    elif method == 'catboost':
        if CatBoostRegressor is None:
            raise ImportError("catboost not installed")
        # train_pool = Pool(data=X_train, label=y_train, cat_features=[0, 3, 5])
        model = CatBoostRegressor(random_seed=random_state, verbose=False)
        param_grid = {
            'iterations': [100, 200],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5]
        }
      
    else:
        raise ValueError(f"Unsupported method '{method}'")
    return model, param_grid



def train_and_tune_model(method, X_train, y_train, search='grid', cv=5, random_state=42, n_iter=10):
    model, param_grid = get_model_and_params(method, random_state)
    start = time.time()
    if param_grid is None or len(param_grid) == 0:
        model.fit(X_train, y_train)
        best_params = None
    else:
        if search == 'grid':
            searcher = GridSearchCV(model, param_grid, cv=cv, scoring='r2')
        else:
            from sklearn.model_selection import RandomizedSearchCV
            searcher = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=n_iter, random_state=random_state, scoring='r2')
        searcher.fit(X_train, y_train)
        model = searcher.best_estimator_
        best_params = searcher.best_params_
    duration = time.time() - start
    return model, duration, best_params



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        'MAE': mean_absolute_error(y_test, y_pred),
    }
    return metrics



if __name__ == "__main__":
    features_path = 'features.csv'   
    labels_path = 'labels.csv'      
    test_size = 0.2
    random_state = 42
    use_scaler = True
    n_iter = 10
    scoring = 'r2'

    method = 'lightgbm'   # 'gbdt', 'xgboost', 'lightgbm', 'catboost'
    """
    Description of Ensemble Learning Regressors and their typical application scenarios
    - GBDT (GradientBoostingRegressor) : classic gradient promotion tree, suitable for medium and small scale data, stable effect.
    - xgboost (XGBRegressor): Efficient implementation, supporting large-scale and sparse data.
    - lightgbm (LGBMRegressor): Fast and memory-friendly, capable of handling high-dimensional sparse data.
    - catboost (CatBoostRegressor): Excellent category feature processing, suitable for data with a large number of category features.
    """

    X_train, X_test, y_train, y_test = prepare_data(features_path, labels_path, test_size, random_state, use_scaler)

    model, train_time, best_params = train_and_tune_model(method, X_train, y_train, search='grid', cv=5, random_state=random_state)

    metrics = evaluate_model(model, X_test, y_test)

    report = {
        'model': method,
        'train_time_sec': train_time,
        'best_params': best_params,
        **metrics
    }

    print("Training and evaluation complete. Report:")
    print(json.dumps(report, indent=4))
