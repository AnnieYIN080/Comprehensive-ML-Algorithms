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

    # This code configures a preprocessing flow that only uses the data standardizer, without using PCA or feature selection.
    use_scaler = True
    # It indicates that a data standardizer will be used, which is typically employed to scale data to a specific range [0,1]
    
    use_pca = False
    # use_pca = False indicates that principal component analysis (PCA) is not used, which is a dimensionality reduction technique
    pca_n_comp = None
    # pca_n_comp = None indicates that if PCA is used, the number of principal components will not be specified, which is consistent with use_pca = False
    # If your goal is to reduce the dimension while retaining information, evaluating the performance stability and cross-validation scores. 
    # Prioritize setting the cumulative interpretation variance threshold corresponding to pca_n_comp (such as 0.90, 0.95).
    
    use_select = False
    # "use_select = False" indicates that feature selection is not used. Feature selection is the process of choosing the most relevant features in the dataset.
    k_best = None
    # k_best = None. If no feature selection is performed, there is no need to specify the number of best features to be retained, which is consistent with use_select = False.
    '''
    1. filtering (Directly evaluate the relationship between each feature and the target variable): Variance screening (Deleting low-variance features), Chi-square test (classification problem)
    from sklearn.pipeline import Pipeline
    # 1.1
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('var_thresh', VarianceThreshold(threshold=0.01)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    # 1.2
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),  # When used with chi2, non-negative data is usually required
        ('selector', SelectKBest(chi2, k=50)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    # 2. wrappering (Use a predictive model to evaluate the effectiveness of a set of features and gradually search for the best subset)
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    
    estimator = LogisticRegression(max_iter=1000)
    selector = RFE(estimator, n_features_to_select=20, step=1)

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('clf', selector),
        ('final', LogisticRegression(max_iter=1000)) 
    ])
    
    # 3. Embedded (Feature selection is carried out simultaneously during the model training process, usually through regularization or importance weighting)
    # 3.1
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = [i for i in indices[:k]]   # Selecting fisrt k Important Features
    
    # 3.2
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=0.01)
    sfm = SelectFromModel(model, threshold='mean', prefit=False)
    sfm.fit(X_train, y_train)
    X_train_selected = sfm.transform(X_train)
    '''
    
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
