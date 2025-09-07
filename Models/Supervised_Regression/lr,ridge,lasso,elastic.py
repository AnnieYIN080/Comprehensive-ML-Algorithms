import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression


def prepare_data(features_path, labels_path, test_size, random_state,
                 use_scaler=True, use_pca=False, pca_n_comp=None, use_select=False, k_best=None):
                   
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()
                   
    if use_scaler:
        X = StandardScaler().fit_transform(X)
    if use_pca and pca_n_comp is not None:
        X = PCA(n_components=pca_n_comp).fit_transform(X)
    if use_select and k_best is not None:
        X = SelectKBest(f_regression, k=k_best).fit_transform(X, y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def build_model(method, alpha=1.0, l1_ratio=0.5, random_state=42):
    if method == 'lr':
        return LinearRegression()
    elif method == 'ridge':
        return Ridge(alpha=alpha, random_state=random_state)
    elif method == 'lasso':
        return Lasso(alpha=alpha, random_state=random_state)
    elif method == 'elastic':
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    else:
        raise ValueError("Unknown method: Supported are 'lr','ridge','lasso','elastic'")


def train_and_tune_model(method, X_train, y_train, search='grid', param_grid=None, cv=5, random_state=42, n_iter=10):
    if param_grid is None:
        param_grid = {}
      
    if method == 'lr':
        model = build_model('lr')
        model.fit(X_train, y_train)
        return model
      
    est = build_model(method, random_state=random_state)
    if search == 'grid':
        model = GridSearchCV(est, param_grid=param_grid, cv=cv)
    else:
        model = RandomizedSearchCV(est, param_distributions=param_grid, cv=cv, random_state=random_state, n_iter=n_iter)
    
    model.fit(X_train, y_train)  
    return model.best_estimator_



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

    method = 'ridge'      # 'lr', 'ridge', 'lasso', 'elastic'
    search = 'grid'       # 'grid', 'random'
    cv = 5
    n_iter = 10

    # Define the parameter grid and adjust it according to the method
    param_grid = {}
    if method in ['ridge', 'lasso']:
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    elif method == 'elastic':
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.9]}

      
    X_train, X_test, y_train, y_test = prepare_data(
        features_path, labels_path, test_size, random_state,
        use_scaler, use_pca, pca_n_comp, use_select, k_best)

    model = train_and_tune_model(
        method, X_train, y_train, search, param_grid, cv, random_state, n_iter)

    results = evaluate_model(model, X_test, y_test)
    print(results)
