#!usr/bin/env python3
'''
# kd algorithm
from sklearn.neighbors import KDTree
tree = KDTree(X)   # Build a KD tree
dist, ind = tree.query(X[:5], k=3)  # Query the three nearest neighbors of the first five sample points
print("KDTree distances:\n", dist)
print("KDTree indices:\n", ind)
'''
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV          # pip install scikit-optimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. load data
def load_data(features_path, labels_path, test_size=test_size, random_state=random_state):
    # load data 
    X = pd.read_csv('features_path').values  # Convert to numpy array
    y = pd.read_csv('labels_path').values.flatten()  # flatten() converts a two-dimensional array to a one-dimensional array
  
    scaler = StandardScaler()    # If the feature scales are significantly different, StandardScaler or MinMaxScaler should be used for standardization.
    X = scaler.fit_transform(X)  # Standardization features
    le = LabelEncoder()    # If the label is a string, it needs to be converted to a numeric value (such as LabelEncoder)
    y = le.fit_transform(y)    # Code the feature
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)



# 2. model choice
def cv_search(k, cv, X_train, y_train):
    # Cross-validation 
  
    cv_scores = []
    k_range = range(1, k)
  
    for k_val in k_range:
        knn = KNeighborsClassifier(n_neighbors=k_val)    # n_neighbors=k  indicating the selection of the k nearest sample points
        scores = cross_val_score(knn, X_train, y_train, cv=cv)  # 5-fold cross-validation, indicating that the dataset is divided into 5 subsets
        cv_scores.append(scores.mean())     # Calculate the average accuracy rate of each k value

    best_idx = np.argmax(cv_scores)       # Select the k value with the highest average accuracy rate
    best_k = list(k_range)[best_idx]
  
    best_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
  
    '''
    results = {
      'best_k': best_k,
      'cv_accuracy': max(cv_scores),
      'method': 'Cross-validation'
      }

    #  Visualize the relationship between the k value and accuracy rate
    import matplotlib.pyplot as plt
    plt.plot(k_range, cv_scores, marker='o')
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Cross-validated accuracy")
    plt.title("Cross-validation - KNN Accuracy vs k")
    plt.grid(True)     # Display grid lines
    plt.show()
    '''
    return best_k, best_model


def grid_search(k, cv, X_train, y_train):
    param_grid = {'n_neighbors': range(1, k)}   # To avoid overfitting caused by excessive k
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv) 
    grid.fit(X_train, y_train)
    '''
    results = {
        'best_params': grid.best_params_,
        'cv_accuracy': grid.best_score_,
        'method': 'GridSearchCV'
     }
     '''
     best_k = grid.best_params_['n_neighbors']
     best_model = grid.best_estimator_  

     return best_k, best_model


def bayes_search(k, n_iter, cv, X_train, y_train, random_state=random_state):
    from skopt import BayesSearchCV
    param_space = {'n_neighbors': (1, k)}
    bs = BayesSearchCV(KNeighborsClassifier(), param_space, n_iter=n_iter, cv=cv, random_state=random_state)    
    # n_iter: Random search count, The greater the number of tests, the more precise they are
    bs.fit(X_train, y_train)
    
    best_k = bs.best_params_['n_neighbors']
    best_model = bs.best_estimator_  
  
    return best_k, best_model
  


def auto_search(k, n_iter, cv, X_train, y_train, random_state=random_state):
    if len(X_train) < 5000:
        return grid_search(k, cv, X_train, y_train)
      
    else:
    # When the data volume is large -> random search
        param_dist = {'n_neighbors': np.arange(1, 3*k, 2)}    # Only consider odd k values to avoid a tie
        search = RandomizedSearchCV(
            KNeighborsClassifier(),
            param_distributions=param_dist,   # Distribution of Sampling Parameters
            n_iter=n_iter,  # Random search count
            cv=cv,
            random_state=random_state
            )
      
    search.fit(X_train, y_train)
    '''
    results = {
        'best_params': search.best_params_,
        'cv_accuracy': search.best_score_,
        'method': 'RandomizedSearchCV' if len(X_train) >= 5000 else 'GridSearchCV'
    }
    '''
    best_model = search.best_estimator_
    best_k = search.best_params_['n_neighbors']
  
    return best_k, best_model



# 3. evaluate the model
def evaluate_model(best_model, X_test, y_test, best_k, method):
    y_pred = best_model.predict(X_test)
    # ROC_AUC may be not suitable enough in KNN
    '''
    y_prob = best_model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovo')
    '''
    return {
        'best_k': best_k,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='macro'),
        'test_recall': recall_score(y_test, y_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_pred, average='macro'),
        # 'test_auc': auc,
        'method': method
    }
  


# 4. use knn model
def Model_KNN(features_path, labels_path, k=k, cv=cv, n_iter=n_iter, method=method, test_size=test_size, random_state=random_state):
    X_train, X_test, y_train, y_test = load_data(features_path, labels_path, test_size, random_state)
  
    """
    KNN: Parameter tuning model evaluation

    Args:
        method (str): Collaborative Optimization
            - 'cv'   : cross validation;   cv_search(k, cv, X_train, y_train)
            - 'grid' : grid search;        grid_search(k, cv, X_train, y_train)
            - 'bs' : bayes search;         bs_search(X_train, y_train, cv, k, n_iter, random_state)
            - 'auto' : automatically choose Grid Search or Randomized Search based on X_train size;   auto_search(X_train, y_train, cv, k, n_iter, random_state)
    Returns:
        dict: Optimal parameter model performance indicators
    """
  
    if method == 'cv':
        best_k, best_model = cv_search(k, cv, X_train, y_train)
    elif method == 'grid':
        best_k, best_model = grid_search(k, cv, X_train, y_train)
    elif method == 'bs':
        best_k, best_model = bayes_search(k, n_iter, cv, X_train, y_train, random_state)
    else:
        best_k, best_model = auto_search(k, n_iter, cv, X_train, y_train, random_state)
      
    return evaluate_model(best_model, X_test, y_test, best_k, method)



# 5. run it 
if __name__ == "__main__":
    results = Model_KNN(
        features_path='features.csv',
        labels_path='labels.csv',
        k=20,
        cv=5,
        n_iter=10,
        method='bs',          # 'cv', 'grid', 'bs', or 'auto'
        test_size=0.2,
        random_state=42,
        # best_k = None,
        # best_model = None
    )
    print(results)
