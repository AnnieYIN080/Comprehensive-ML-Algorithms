# Machine Learning <English version>
## A. Supervised Learning Model
### 1. Classification model (output for discrete label prediction) :
    K-nearest neighbor (K-NN) ‌, Decision Tree, Random Forest, support vector machine(SVM)‌‌, Naive Bayes(Naive Bayes)‌, Logistic Regression(Logistic Regression), Neural Networks
### 2. Regression model (output for continuous value prediction):
    Decision Tree Regression ‌, Random Forest Regression ‌, Support Vector Regression(SVR)‌, Gradient Boosting Regression, Linear Regression, Polynomial Regression, Ridge Regression, Lasso Regression, Elastic Net Regression‌, XGBoost Regression‌, LightGBM Regression‌, CatBoost Regression ‌
## B. Unsupervised Learning Models
### 1. Clustering model ‌ (data grouping) :
    K-Means ‌, Hierarchical Clustering ‌, DBSCAN (density-based clustering) ‌
### 2. Dimensionality reduction model ‌ (feature extraction) :
    Principal component analysis (PCA) ‌, t-SNE (visualization of high-dimensional data) ‌, Autoencoder ‌
### ‌ 3. Association rule learning ‌ :
    Apriori algorithm (market basket analysis)‌

Supervised learning is applicable to scenarios with clear output requirements. All supervised learning models (classification, regression) can use the provided three methods to find the optimal parameters.<br>
**Suggestion:** When the dataset is too large, grid Search is too slow. You can try Random Search or Bayesian Optimization, especially Deep Learning

# ================= explains =================
1. Data Preparation: Ensure that the paths of feature and tag files are correct and the data format meets the requirements.<br>
2. Data Preprocessing: Select an appropriate preprocessing method based on the characteristics of the data.<br>
3. The model choice: choose the appropriate supervision and learning model according to task demand, here can be used [H2O]([url](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)) instead.<br>
4. Parameter tune: Optimize model parameters using methods such as cross-validation and grid search.<br>
5. Model Evaluation: Evaluate the model performance using metrics such as accuracy rate, precision rate, recall rate, and F1 score.<br>
6. Result Interpretation: Interpret the model results based on business requirements and provide actionable suggestions.<br>
7. Continuous Improvement: Continuously optimize the model and data processing procedures based on model performance and new data.<br>



# 机器学习基础 《中文版本》
## 一、监督学习模型
### ‌1. 分类模型‌（输出用于离散标签预测）：
    K近邻（K-NN）‌，决策树（Decision Tree），随机森林（Random Forest），支持向量机（SVM）‌‌，朴素贝叶斯（Naive Bayes）‌，逻辑回归（Logistic Regression）‌，神经网络（Neural Networks）
### ‌2. 回归模型‌（输出用于连续值预测）：
    决策树回归（Decision Tree Regression）‌，随机森林回归（Random Forest Regression）‌，支持向量回归（SVR）‌，梯度提升回归（Gradient Boosting Regression），线性回归（Linear Regression）‌，多项式回归（Polynomial Regression）‌，岭回归（Ridge Regression）‌，Lasso 回归 (Lasso Regression)‌，弹性网回归（Elastic Net Regression）‌，XGBoost 回归（XGBoost Regression）‌，LightGBM 回归（LightGBM Regression‌），CatBoost 回归（CatBoost Regression）‌
## 二、无监督学习模型
###  ‌1. 聚类模型‌（数据分组）：
    K均值（K-Means）‌，层次聚类（Hierarchical Clustering）‌，DBSCAN（基于密度的聚类）‌
###  2. 降维模型‌（特征提取）：
    主成分分析（PCA）‌，t-SNE（可视化高维数据）‌，自编码器（Autoencoder）‌
### ‌ 3. 关联规则学习‌：
    Apriori算法（市场篮分析）‌

监督学习适用于有明确输出需求的场景, 所有监督学习模型（分类、回归）都能用提供的三种方法去找最优参数。<br>

**建议：** 数据集太大时，网格搜索太慢，可以尝试随机搜索（Random Search）或贝叶斯优化（Bayesian Optimization），尤其是Deep Learning<br>

# 说明
1.数据准备：确保特征和标签文件路径正确，数据格式符合要求。<br>
2.数据预处理：根据数据特点，选择合适的预处理方法。<br>
3.模型选择：根据任务需求选择合适的监督学习模型，此处可以用[H2O]([url](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html))代替。<br>
4.参数调优(parameters tune)：使用交叉验证和网格搜索等方法优化模型参数。<br>
5.模型评估：使用准确率、精确率、召回率、F1分数等指标评估模型性能。<br>
6.结果解释：根据业务需求解释模型结果，提供可操作的建议。<br>
7.持续改进：根据模型表现和新数据，持续优化模型和数据处理流程。<br>
