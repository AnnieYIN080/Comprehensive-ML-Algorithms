#!usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV          # pip install scikit-optimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(features_path, labels_path, test_size=test_size, random_state=random_state):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()
    X = StandardScaler().fit_transform(X)
    y = LabelEncoder().fit_transform(y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def cv_search(n_estimators_start, n_estimators_end, cv, X_train, y_train, random_state):
    cv_scores = []
    est_range = range(n_estimators_start, n_estimators_end) 
    for n in est_range:
        rf = RandomForestClassifier(n_estimators=n, random_state=random_state)
        scores = cross_val_score(rf, X_train, y_train, cv=cv)
        cv_scores.append(scores.mean())
    best_idx = np.argmax(cv_scores)
    best_n = list(est_range)[best_idx]
    best_model = RandomForestClassifier(n_estimators=best_n, random_state=random_state).fit(X_train, y_train)
    return best_n, best_model


def grid_search(n_estimators_start, n_estimators_end cv, X_train, y_train, random_state):
    param_grid = {'n_estimators': range(n_estimators_start, n_estimators_end)}
    grid = GridSearchCV(RandomForestClassifier(random_state=random_state), param_grid, cv=cv)
    grid.fit(X_train, y_train)
    best_n = grid.best_params_['n_estimators']
    best_model = grid.best_estimator_
    return best_n, best_model


def bayes_search(n_estimators_start, n_estimators_end, n_iter, cv, X_train, y_train, random_state):
    param_space = {'n_estimators': (n_estimators_start, n_estimators_end)}   
    bs = BayesSearchCV(RandomForestClassifier(random_state=random_state), param_space, n_iter=n_iter, cv=cv, random_state=random_state)
    bs.fit(X_train, y_train)
    best_n = bs.best_params_['n_estimators']
    best_model = bs.best_estimator_
    return best_n, best_model


def auto_search(n_estimators_start, n_estimators_end, n_iter, cv, X_train, y_train, random_state):
    if len(X_train) < 5000:
        return grid_search(n_estimators_limit, cv, X_train, y_train)
    else:
        param_dist = {'n_estimators': np.arange(n_estimators_start, n_estimators_end*3, 10)}
        search = RandomizedSearchCV(RandomForestClassifier(random_state=random_state), param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state)
        search.fit(X_train, y_train)
        best_n = search.best_params_['n_estimators']
        best_model = search.best_estimator_
        return best_n, best_model


def evaluate_model(best_model, X_test, y_test, best_n, method):
    y_pred = best_model.predict(X_test)
   
    y_prob = best_model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovo')
    
    return {
        'best_n_estimators': best_n,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='macro'),
        'test_recall': recall_score(y_test, y_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_pred, average='macro'),
        'test_auc': auc,
        'method': method
    }


def Model_RF(features_path, labels_path, n_estimators_limit=n_estimators_limit, cv=cv, n_iter=n_iter, method=method, test_size=test_size, random_state=random_state):
    X_train, X_test, y_train, y_test = load_data(features_path, labels_path, test_size, random_state)
    if method == 'cv':
        best_n, best_model = cv_search(n_estimators_start, n_estimators_end, cv, X_train, y_train, random_state)
    elif method == 'grid':
        best_n, best_model = grid_search(n_estimators_start, n_estimators_end, cv, X_train, y_train, random_state)
    elif method == 'bs':
        best_n, best_model = bayes_search(n_estimators_start, n_estimators_end, n_iter, cv, X_train, y_train, random_state)
    else:
        best_n, best_model = auto_search(n_estimators_start, n_estimators_end, n_iter, cv, X_train, y_train, random_state)
    return evaluate_model(best_model, X_test, y_test, best_n, method)



if __name__ == "__main__":
    results = Model_RF(
        features_path='features.csv',
        labels_path='labels.csv',
        n_estimators_start=10, 
        n_estimators_end=100,
        cv=5,
        n_iter=10,
        method='bs',  # 'cv', 'grid', 'bs', or 'auto'
        test_size=0.2,
        random_state=42
    )
    print(results)
