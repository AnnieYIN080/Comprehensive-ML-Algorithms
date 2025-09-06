#!usr/bin/env python3
# using Multilayer Perceptron of Neural Network 

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV          # pip install scikit-optimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import label_binarize

def load_data(features_path, labels_path, test_size, random_state):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()
    X = StandardScaler().fit_transform(X)  # MLP is sensitive to feature scales
    y = LabelEncoder().fit_transform(y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def cv_search(X_train, y_train, cv, param_grid, max_iter, random_state):
    cv_scores = []
    # Search for the size of hidden layers
    for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, cv=cv)
        cv_scores.append((hidden_layer_sizes, scores.mean()))
    best_hidden_layer = max(cv_scores, key=lambda x: x[1])[0]
    best_model = MLPClassifier(hidden_layer_sizes=best_hidden_layer, max_iter=max_iter, random_state=random_state).fit(X_train, y_train)
    return best_hidden_layer, best_model

def grid_search(X_train, y_train, cv, param_grid, max_iter, random_state):
    grid = GridSearchCV(MLPClassifier(max_iter=max_iter, random_state=random_state), param_grid, cv=cv)
    grid.fit(X_train, y_train)
    return grid.best_params_['hidden_layer_sizes'], grid.best_estimator_

def bayes_search(X_train, y_train, cv, n_iter, param_space, max_iter, random_state):
    bs = BayesSearchCV(MLPClassifier(max_iter=max_iter, random_state=random_state), param_space, n_iter=n_iter, cv=cv, random_state=random_state)
    bs.fit(X_train, y_train)
    return bs.best_params_['hidden_layer_sizes'], bs.best_estimator_

def auto_search(X_train, y_train, cv, n_iter, param_grid, param_dist, max_iter, random_state):
    if len(X_train) < 5000:
        return grid_search(X_train, y_train, cv, param_grid)
    else:
        search = RandomizedSearchCV(MLPClassifier(max_iter=max_iter, random_state=random_state), param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state)
        search.fit(X_train, y_train)
        return search.best_params_['hidden_layer_sizes'], search.best_estimator_

def evaluate_model(best_model, X_test, y_test, best_param, method):
    y_pred = best_model.predict(X_test)
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovo')
    else:
        auc = None
    return {
        'best_hidden_layer_sizes': best_param,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='macro'),
        'test_recall': recall_score(y_test, y_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_pred, average='macro'),
        'test_auc': auc,
        'method': method
    }

if __name__ == "__main__":
    features_path = 'features.csv'
    labels_path = 'labels.csv'
    test_size = 0.2
    random_state = 42
    cv = 5
    n_iter = 10
    max_iter = 500
    method = 'bs'  # 'cv', 'grid', 'bs', 'auto'

    param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)]}
    param_dist = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)]}
    param_space = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)]}  
    # Bayesian optimization is rather difficult to directly use tuples and requires workarounds such as strings

  
    X_train, X_test, y_train, y_test = load_data(features_path, labels_path, test_size, random_state)

    if method == 'cv':
        best_param, best_model = cv_search(X_train, y_train, cv, param_grid, max_iter, random_state)
    elif method == 'grid':
        best_param, best_model = grid_search(X_train, y_train, cv, param_grid, max_iter, random_state)
    elif method == 'bs':
        best_param, best_model = bayes_search(X_train, y_train, cv, n_iter, param_space, max_iter, random_state)
    else:
        best_param, best_model = auto_search(X_train, y_train, cv, n_iter, param_grid, param_dist, max_iter, random_state)

    results = evaluate_model(best_model, X_test, y_test, best_param, method)
    print(results)
