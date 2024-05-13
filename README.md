# Predicting Heart Failure with Machine Learning

## Description
This Jupyter Notebook contains the code for predicting heart failure using machine learning models. The dataset used for this project is the Heart Failure Clinical Records dataset, which includes various clinical features of patients such as age, sex, smoking status, and medical history. The goal is to predict the likelihood of a patient experiencing a heart failure event based on these features.

## Steps
1. **Installing Dependencies**: The code begins by installing the necessary libraries, including `kaggle`, and setting up the Kaggle API for downloading the dataset.

2. **Data Preparation**: The dataset is downloaded and loaded into a pandas DataFrame. It is then inspected using `df.head()`, `df.describe()`, and `df.info()` to understand its structure and contents.

3. **Exploratory Data Analysis (EDA)**: Various visualizations such as scatter plots, bar plots, and heatmaps are created to explore the relationships between different features and the target variable (`DEATH_EVENT`).

4. **Data Preprocessing**: Some features are dropped from the dataset based on correlation analysis (`df.corr()`) and domain knowledge. The dataset is split into training and testing sets using `train_test_split()`.

5. **Model Selection**: Several machine learning models are evaluated, including Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Naive Bayes, Support Vector Machine, AdaBoost, and XGBoost. The models are trained and evaluated using cross-validation (`StratifiedKFold`) to ensure robust performance metrics.

6. **Model Evaluation**: The models are evaluated using accuracy scores and confusion matrices. The Random Forest Classifier is selected as the final model due to its high accuracy.

7. **Hyperparameter Tuning**: Grid search (`GridSearchCV`) is performed to find the best hyperparameters for the Random Forest Classifier, further improving its performance.

8. **Performance Visualization**: The accuracy of the Random Forest model is plotted against the number of trees (`n_estimators`) to visualize its performance.

## Results
The Random Forest Classifier achieved an accuracy of 95% in predicting heart failure events, making it a reliable model for predicting heart failure based on the given clinical features.

