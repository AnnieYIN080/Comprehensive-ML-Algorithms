import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError:
    GradientBoostingClassifier = None
  
# https://xgboost.readthedocs.io/en/stable/install.html
# $ pip install xgboost  # or  pip install --user xgboost or pip install xgboost-cpu
# $ conda install -c conda-forge py-xgboost=*=cpu*    # CPU
# $ export CONDA_OVERRIDE_CUDA="12.8"
# $ conda install -c conda-forge py-xgboost=*=cuda*    # NVIDIA GPU
# Install according to your environment. Examples use common package names
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
  
# https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
# $ pip install lightgbm
try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except ImportError:
    LGBMClassifier = None

# https://catboost.ai/docs/en/installation/python-installation-method-pip-install
# $ pip install catboost
# $ pip install ipywidgets    # visualization tools
# $ jupyter nbextension enable --py widgetsnbextension  # Turn on the widgets extension
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

def prepare_data(features_path, labels_path, test_size, random_state, use_scaler=True):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()
    if use_scaler:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_ensemble_model_and_params(method, random_state=42):
    param_grid = {}
    if method == 'gbdt':
        if GradientBoostingClassifier is None:
            raise ImportError("scikit-learn GradientBoostingClassifier not installed")
        
        model = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
      
    elif method == 'xgboost':
        if XGBClassifier is None:
            raise ImportError("xgboost not installed")
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
      
    elif method == 'lightgbm':
        if LGBMClassifier is None:
            raise ImportError("lightgbm not installed")
        model = LGBMClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [-1, 5, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'feature_fraction': [0.8, 1.0]
        }

  
    elif method == 'catboost':
        if CatBoostClassifier is None:
            raise ImportError("catboost not installed")
        model = CatBoostClassifier(random_seed=random_state, verbose=False)
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
    model, param_grid = get_ensemble_model_and_params(method, random_state)
    start = time.time()
    if param_grid is None:
        model.fit(X_train, y_train)
        best_params = None
      
    else:
        if search == 'grid':
            searcher = GridSearchCV(model, param_grid, cv=cv)
        else:
            from sklearn.model_selection import RandomizedSearchCV
            searcher = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=n_iter, random_state=random_state)
        
        searcher.fit(X_train, y_train)
        model = searcher.best_estimator_
        best_params = searcher.best_params_
      
    duration = time.time() - start
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

    method = 'lightgbm'   # 'gbdt'、'xgboost'、'lightgbm'、'catboost'
    # Model and Scene Annotations
    # GBDT: Classic Gradient Boosting Tree, stable, suitable for small and medium-sized datasets, non-high-dimensional sparse data. It is often used for regression, classification and sorting.
    # XGBoost: Efficient implementation, supports sparse data and parallelism, suitable for big data and high-dimensional data.
    # LightGBM: Faster and more memory-efficient, supports large-scale datasets and categorical features, suitable for high-dimensional sparse data.
    # CatBoost: Built-in category feature processing, friendly to handling category data, suitable for tasks with a large number of category features.

    X_train, X_test, y_train, y_test = prepare_data(features_path, labels_path, test_size, random_state, use_scaler)

    model, train_time, best_params = train_and_tune_model(method, X_train, y_train, search='grid', cv=5, random_state=random_state)

    metrics = evaluate_model(model, X_test, y_test)

    report = {
        'model': method,
        'train_time_sec': train_time,
        'best_params': best_params,
        **metrics
    }

    print("Training complete, evaluation report:")
    print(json.dumps(report, indent=4))
