#!usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV          # pip install scikit-optimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import label_binarize

# 1. load data
def load_data(features_path, labels_path, test_size, random_state):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.flatten()
    # For Naive Bayes, feature standardization is optional and is not mandatory here
    # X = StandardScaler().fit_transform(X)  
    y = LabelEncoder().fit_transform(y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# 2. Cross-validation parameter tuning example (can ignore here, NB no need 
def cv_search(cv, X_train, y_train):
    model = GaussianNB()
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    model.fit(X_train, y_train)
    return model, scores.mean()

# 3. Evaluate the model and supplement with AUC 
def evaluate_model(best_model, X_test, y_test, method):
    y_pred = best_model.predict(X_test)
    # Calculate the multi-classification probability output
    y_prob = best_model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    # Calculate multi-class AUC, ovo strategy
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovo')
    
    return {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='macro'),
        'test_recall': recall_score(y_test, y_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_pred, average='macro'),
        'test_auc': auc,
        'method': method
    }

# 4. Main model function
def Model_NB(features_path, labels_path, cv, test_size, random_state):
    X_train, X_test, y_train, y_test = load_data(features_path, labels_path, test_size, random_state)
    best_model, cv_score = cv_search(cv, X_train, y_train)
    print(f"Cross-validation accuracy: {cv_score:.4f}")
    return evaluate_model(best_model, X_test, y_test, 'cv')

# 5. run demo
if __name__ == "__main__":
    results = Model_NB(
        features_path='features.csv',
        labels_path='labels.csv',
        cv=5,
        test_size=0.2,
        random_state=42
    )
    print(results)
