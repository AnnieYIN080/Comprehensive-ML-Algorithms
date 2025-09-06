# Machine Learning <English version>

# Steps
1. Data Preparation: Ensure that the paths of feature and tag files are correct and the data format meets the requirements.<br>
2. Data Preprocessing: Select an appropriate preprocessing method based on the characteristics of the data.<br>
3. The model choice: choose the appropriate supervision and learning model according to task demand, here can be used [H2O]([url](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)) instead.<br>
4. Parameter tune: Optimize model parameters using methods such as cross-validation and grid search.<br>

Algorithm parameter tuning refers to the process in machine learning or data analysis where the parameters of an algorithm are adjusted to minimize the loss function (such as loss in SSDS) in order to optimize the performance of the algorithm. This typically involves running the algorithm on the training dataset, evaluating its performance (such as accuracy, recall rate, etc.), then adjusting the parameters based on the evaluation results, and repeating this process until the best parameter combination is found to achieve the best results for the algorithm on a specific task.<br>

5. Model Evaluation: Evaluate the model performance using metrics such as accuracy rate, precision rate, recall rate, and F1 score.<br>
6. Result Interpretation: Interpret the model results based on business requirements and provide actionable suggestions.<br>
7. Continuous Improvement: Continuously optimize the model and data processing procedures based on model performance and new data.<br>


## A. Supervised Learning Model
### 1. Classification model (output for discrete label prediction)
    K-nearest neighbor (K-NN) ‌, Naive Bayes(Naive Bayes)‌,  Logistic Regression(Logistic Regression), 
    Decision Tree, Random Forest, support vector machine(SVM)‌‌, Neural Networks
### 2. Regression model (output for continuous value prediction)
    Decision Tree Regression ‌, Random Forest Regression ‌, Support Vector Regression(SVR)‌, 
    Linear Regression, Polynomial Regression, Ridge Regression, Lasso Regression, Elastic Net Regression‌, 
    # Gradient Boosting Regression, XGBoost Regression‌, LightGBM Regression‌, CatBoost Regression ‌
## B. Unsupervised Learning Models
### 1. Clustering model ‌ (data grouping) 
    K-Means ‌, Hierarchical Clustering ‌, DBSCAN (density-based clustering) ‌
### 2. Dimensionality reduction model ‌ (feature extraction) 
    Principal component analysis (PCA) ‌, t-SNE (visualization of high-dimensional data) ‌, Autoencoder ‌
### ‌ 3. Association rule learning ‌ 
    Apriori algorithm (market basket analysis)‌

## C. Ensemble learning Models
    Bagging (bootstrap aggregation)，Boosting， VotingClassifier，StackingClassifier
    # Gradient Boosting Regression, XGBoost Regression‌, LightGBM Regression‌, CatBoost Regression 



# Supervised Learning: Classification Models

## 1. K-Nearest Neighbors (KNN)
K-Nearest Neighbors (KNN) is an instance-based supervised learning algorithm that performs **classification** based on the distance between samples (rarely used for regression).  
For a new sample, KNN finds the K closest neighbors in the training set and predicts the class or value based on these neighbors.  
Advantages of KNN include simplicity, ease of implementation, and the ability to handle multi-class problems. Disadvantages include high computational complexity, especially on large-scale datasets, and sensitivity to noise.

## 2. Decision Tree
Decision Tree is a tree-structured model used for both **classification and regression**, which recursively partitions data to construct a tree representing the decision process.  
Each node tests a feature, branches represent different test outcomes, and leaf nodes provide class or value predictions.  
Advantages include simplicity, interpretability, and ability to handle nonlinear relationships. Disadvantages include tendency to overfit, especially when trees are very deep.

## 3. Random Forest
Random Forest is an **ensemble learning method** that constructs multiple decision trees for **classification or regression**, combining their predictions to improve accuracy and robustness.  
Each tree is trained using random subsets of features and samples, reducing overfitting risk.  
Advantages include handling high-dimensional data and strong noise resistance, while drawbacks include a more complex model structure and difficulty of interpretability.

## 4. Support Vector Machine (SVM)
Support Vector Machine (SVM) is a supervised learning algorithm mainly used for **classification and regression**.  
The core idea is to find an optimal hyperplane that separates different classes while maximizing the margin between classes.  
For linearly inseparable data, SVM employs kernel functions to map data into higher-dimensional space for linear separation.  
Advantages of SVM include handling high-dimensional data and strong generalization ability. Disadvantages are sensitivity to parameter and kernel function selection.

## 5. Naive Bayes
Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem, assuming feature independence. It is mainly used for **classification tasks**.  
It predicts the class of samples by computing prior probabilities and conditional probabilities of each class.  
Naive Bayes is suitable for text classification, spam filtering, and similar tasks.  
Its advantages are high computational efficiency and scalability to large datasets; disadvantages include the strong independence assumption which may affect classification performance.

## 6. Logistic Regression
Logistic Regression is a generalized **linear model** primarily used for **binary and multi-class classification**.  
It predicts the probability of class membership by linearly combining input features and applying the logistic (sigmoid) function.  
Applications include credit scoring and medical diagnosis.  
Advantages are simplicity and interpretability, but it performs best when data is linearly separable and may underperform on nonlinear data.

## 7. Neural Networks
Neural Networks are computational models inspired by biological neural systems, consisting of multiple layers of neurons (nodes). They are used for **classification and regression tasks**.  
Each neuron receives input signals, applies weighted sums and biases, then outputs through activation functions.  
Neural networks learn complex relationships between input data and output labels by adjusting weights and biases.  
Advantages include the ability to capture complex **nonlinear relationships**, useful in image recognition, natural language processing, etc. Drawbacks include long training times and susceptibility to overfitting.


# Supervised Learning: Regression Models

## 11. Linear Regression
Linear Regression is a fundamental **regression analysis** method used to model the linear relationship between input features and continuous output variables.  
It estimates model parameters by minimizing the mean squared error between predicted and true values.  
Linear regression is commonly applied to tasks such as house price prediction and sales forecasting.  
Its advantages include simplicity, interpretability, and computational efficiency; however, it assumes a strong linear relationship and may fail to capture complex **nonlinear relationships**.

## 12. Polynomial Regression
Polynomial Regression is an **extension of linear regression** that captures **nonlinear relationships** between input features and output variables by introducing polynomial features.  
It transforms the input features into polynomial terms, then fits a linear regression model. Polynomial regression is suitable for modeling complex nonlinear relationships.  
Its advantage is modeling nonlinearity, while it is prone to overfitting, especially with high-degree polynomials.

## 13. Ridge Regression
Ridge Regression is a regularized **linear regression** method that introduces an L2 penalty term in the loss function to prevent overfitting.  
By controlling the size of model parameters, it produces smoother models with better generalization.  
Ridge regression is effective for multicollinearity and high-dimensional data modeling.  
Its advantage is improved model stability and handling multicollinearity; its disadvantage is the need to choose proper regularization parameters.

## 14. Lasso Regression
Lasso Regression is a regularized **linear regression** method that uses an L1 penalty term in the loss function to prevent overfitting.  
It controls model parameters' magnitude and performs feature selection by shrinking less important coefficients to zero.  
Lasso regression is suitable for high-dimensional data and feature selection.  
Its advantages are effective high-dimensional handling and feature selection, improving model interpretability; its drawback is the requirement to select appropriate regularization parameters.

## 15. Elastic Net Regression
Elastic Net Regression combines Ridge and Lasso regression by integrating both L1 and L2 regularization terms in the loss function to prevent overfitting.  
By adjusting the weights of the two penalty terms, it can simultaneously perform feature selection and model smoothing.  
Elastic Net is suitable for high-dimensional data modeling and feature selection.  
Its advantages include handling high dimensions and feature selection; its disadvantage is the need for careful parameter tuning.

## 16. Decision Tree Regression
Decision Tree Regression is a tree-based **regression** method that recursively partitions data into different regions and predicts outputs using simple models (e.g., constant values) within each region.  
It chooses the best feature and split point to minimize prediction error. Decision tree regression is suitable for modeling **nonlinear relationships**.  
Its advantages include interpretability and ability to model nonlinear relationships; its drawbacks include tendency to overfit, especially with deep trees.

## 17. Random Forest Regression
Random Forest Regression is an ensemble regression method that combines predictions from multiple decision trees to improve performance. It is suitable for modeling **nonlinear relationships**.  
Random Forest employs bootstrap sampling and trains multiple trees whose predictions are averaged for the final result.  
Advantages include ability to model nonlinearity and improved accuracy and robustness; disadvantage is model complexity and longer training time.

## 18. Support Vector Regression (SVR)
Support Vector Regression (SVR) is a regression method based on support vector machines, which seeks an optimal hyperplane in a high-dimensional feature space that fits the input features to continuous output variables.  
SVR introduces an ε-insensitive loss function to tolerate small errors, enhancing generalization. SVR is suitable for modeling complex **nonlinear relationships**.  
Advantages include strong nonlinear modeling and generalization ability; disadvantages include longer training time and complex parameter selection.


# Ensemble Models

Ensemble learning is a method that improves model performance by combining the predictions of multiple base learners. It is mainly used for **classification and regression** tasks.  
Common ensemble learning methods include Bagging (Bootstrap Aggregating) and Boosting.  
Bagging trains multiple base learners (independent models of the same type) **in parallel** by sampling training data with replacement, and obtains the final prediction by voting or averaging the outputs of these models (e.g., voting for classification, averaging for regression).  
Boosting trains multiple base learners **sequentially** (dependent on each other), where each subsequent model corrects the errors of the previous one. Boosting focuses more on misclassified samples from previous rounds to gradually improve accuracy.  
Stacking trains multiple independent base learners of different types **in parallel**, then trains a meta-model to combine their predictions.  
The advantage of ensemble learning is improved accuracy and robustness; the downside is increased model complexity and longer training time.

## 1. Bagging (Bootstrap Aggregation)

Bagging aggregates the predictions of all base models by averaging, which effectively reduces high variance (overfitting).  
Thus, base models in Bagging are often allowed to overfit individually.  

## 2. Boosting

Boosting is often used to improve models with high bias (underfitting).  
Typical models include XGBoost, AdaBoost, and Gradient Tree Boosting (GBDT: Gradient Boosting Decision Tree).

## 3. VotingClassifier

VotingClassifier is an ensemble method that combines the predictions of multiple different base learners to improve model performance, mainly for **classification** tasks.  
It supports two voting strategies: hard voting and soft voting.  
Hard voting determines the final class based on majority votes of the base learners, while soft voting uses weighted averages of predicted probabilities to make decisions.  
Advantages include combining strengths of multiple different classifiers to improve overall performance. The disadvantage is increased complexity and training time.

## 4. StackingClassifier

StackingClassifier is an ensemble method that combines predictions of multiple independent and different types of base learners, mainly used for **classification**.  
Unlike VotingClassifier, StackingClassifier uses a meta-learner to learn from the base learners’ predictions and generate final outputs.  
The meta-learner is usually a simple model such as logistic regression or linear regression.  
Advantages include leveraging multiple model types’ strengths to boost overall performance; disadvantages include higher complexity and longer training time.




