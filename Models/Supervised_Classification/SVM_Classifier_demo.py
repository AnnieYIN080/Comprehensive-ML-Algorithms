#!usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV          # pip install scikit-optimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(features_path, labels_path, test_size, random_state):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()
    X = StandardScaler().fit_transform(X)    # Standardized features, SVM is sensitive to feature scale
    y = LabelEncoder().fit_transform(y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def cv_search(X_train, y_train, cv, param_grid):
    cv_scores = []
    param_values = param_grid['C']
    for val in param_values:
        model = SVC(C=val, kernel='rbf')
        scores = cross_val_score(model, X_train, y_train, cv=cv)
        cv_scores.append(scores.mean())
    best_val = param_values[np.argmax(cv_scores)]
    best_model = SVC(C=best_val, kernel='rbf').fit(X_train, y_train)
    return best_val, best_model


def grid_search(X_train, y_train, cv, param_grid):
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=cv)
    grid.fit(X_train, y_train)
    return grid.best_params_['C'], grid.best_estimator_


def bayes_search(X_train, y_train, cv, n_iter, param_space, random_state):
    bs = BayesSearchCV(SVC(kernel='rbf'), param_space, n_iter=n_iter, cv=cv, random_state=random_state)
    bs.fit(X_train, y_train)
    return bs.best_params_['C'], bs.best_estimator_

def auto_search(X_train, y_train, cv, n_iter, param_grid, param_dist, random_state):
    if len(X_train) < 5000:
        return grid_search(X_train, y_train, cv, param_grid)
    else:
        search = RandomizedSearchCV(SVC(kernel='rbf'), param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state)
        search.fit(X_train, y_train)
        return search.best_params_['C'], search.best_estimator_

def evaluate_model(best_model, X_test, y_test, best_C, method):
    y_pred = best_model.predict(X_test)
    
    y_prob = best_model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
   
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovo')
    
    return {
        'best_C': best_C,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='macro'),
        'test_recall': recall_score(y_test, y_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_pred, average='macro'),
        'test_auc': auc,
        'method': method
    }

def Model_SVM(features_path, labels_path, cv, n_iter, param_grid, param_space, param_dist, method, test_size, random_state):
    # Data loading
    X_train, X_test, y_train, y_test = load_data(features_path, labels_path, test_size, random_state)
    
    # Call the corresponding function according to the selected parameter tuning method
    if method == 'cv':
        best_C, best_model = cv_search(X_train, y_train, cv, param_grid)
    elif method == 'grid':
        best_C, best_model = grid_search(X_train, y_train, cv, param_grid)
    elif method == 'bs':
        best_C, best_model = bayes_search(X_train, y_train, cv, n_iter, param_space, random_state)
    else:
        best_C, best_model = auto_search(X_train, y_train, cv, n_iter, param_grid, param_dist, random_state)

    # Evaluating
    return evaluate_model(best_model, X_test, y_test, best_C, method)


if __name__ == "__main__":
  results = Model_SVM(
    # Unified definition of parameters
    features_path = 'features.csv'
    labels_path = 'labels.csv'
    test_size = 0.2
    random_state = 42
    cv = 5
    n_iter = 10
    method = 'bs'  # 'cv', 'grid', 'bs', 'auto'
    
    # The parameter space for parameter adjustment
    param_grid = {'C': [0.1, 1, 10, 100, 1000]}  # Grid search parameters
    param_dist = {'C': np.logspace(-3, 3, 20)}   # Random search parameters
    param_space = {'C': (0.001, 1000)}           # Bayesian optimization parameters, continuous interval
  )
  print(results)
