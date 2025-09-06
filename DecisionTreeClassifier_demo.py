#!usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV          # pip install scikit-optimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. load data
def load_data(features_path, labels_path, test_size=test_size, random_state=random_state):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()

    # Decision trees are usually not sensitive to normalization, but this is retained for convenience and universality
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# 2. model choice
def cv_search(max_depth_limit, cv, X_train, y_train):
    cv_scores = []
    depth_range = range(1, max_depth_limit)
    for depth in depth_range:
        dt = DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(dt, X_train, y_train, cv=cv)
        cv_scores.append(scores.mean())
    best_idx = np.argmax(cv_scores)
    best_depth = list(depth_range)[best_idx]
    best_model = DecisionTreeClassifier(max_depth=best_depth).fit(X_train, y_train)
    return best_depth, best_model


def grid_search(max_depth_limit, cv, X_train, y_train):
    param_grid = {'max_depth': range(1, max_depth_limit)}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cv)
    grid.fit(X_train, y_train)
    best_depth = grid.best_params_['max_depth']
    best_model = grid.best_estimator_
    return best_depth, best_model


def bayes_search(max_depth_limit, n_iter, cv, X_train, y_train, random_state=random_state):
    param_space = {'max_depth': (1, max_depth_limit)}
    bs = BayesSearchCV(DecisionTreeClassifier(), param_space, n_iter=n_iter, cv=cv, random_state=random_state)
    bs.fit(X_train, y_train)
    best_depth = bs.best_params_['max_depth']
    best_model = bs.best_estimator_
    return best_depth, best_model


def auto_search(max_depth_limit, n_iter, cv, X_train, y_train, random_state=random_state):
    if len(X_train) < 5000:
        return grid_search(max_depth_limit, cv, X_train, y_train)
    else:
        param_dist = {'max_depth': np.arange(1, max_depth_limit * 3, 2)}
        search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state)
        search.fit(X_train, y_train)
        best_depth = search.best_params_['max_depth']
        best_model = search.best_estimator_
        return best_depth, best_model


# 3. evaluate the model
def evaluate_model(best_model, X_test, y_test, best_depth, method):
    y_pred = best_model.predict(X_test)
    return {
        'best_max_depth': best_depth,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='macro'),
        'test_recall': recall_score(y_test, y_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_pred, average='macro'),
        'method': method
    }


# 4. use decision tree model
def Model_DT(features_path, labels_path, max_depth_limit=max_depth_limit, cv=cv, n_iter=n_iter, method=method, test_size=test_size, random_state=random_state):
    X_train, X_test, y_train, y_test = load_data(features_path, labels_path, test_size, random_state)
    
    if method == 'cv':
        best_depth, best_model = cv_search(max_depth_limit, cv, X_train, y_train)
    elif method == 'grid':
        best_depth, best_model = grid_search(max_depth_limit, cv, X_train, y_train)
    elif method == 'bs':
        best_depth, best_model = bayes_search(max_depth_limit, n_iter, cv, X_train, y_train, random_state)
    else:
        best_depth, best_model = auto_search(max_depth_limit, n_iter, cv, X_train, y_train, random_state)
    
    return evaluate_model(best_model, X_test, y_test, best_depth, method)



# 5. run it 
if __name__ == "__main__":
    results = Model_DT(
        features_path='features.csv',
        labels_path='labels.csv',
        max_depth_limit=20,   # The hyperparameter space becomes the maximum depth range of the decision tree.
        cv=5,
        n_iter=10,
        method='bs',        # 'cv', 'grid', 'bs', or 'auto'
        test_size=0.2,
        random_state=42
    )
    print(results)
