# Telco Customer Churn

In this project of the IE500 Data Mining Course at the University of Mannheim, we are analysing the Telco Customer Churn dataset.

## Project Overview

This project involves analyzing the Telco Customer Churn dataset to understand customer behavior and identify factors influencing churn. The analysis is structured into several steps, each documented in separate Jupyter notebooks.

## Steps in the Preprocessing

### 1. Data Loading
We begin by loading the datasets into the workspace. This includes both the Kaggle dataset and additional datasets from IBM. The data is read from Excel files and saved as CSV files for easier manipulation.

### 2. Dataset Integration
Next, we combine the relevant datasets into a single, unified dataset. This step ensures that all necessary information is consolidated for subsequent analysis.

### 3. Handling Missing Values
We identify and address missing values in the dataset to ensure data integrity. This step includes analyzing the extent of missing data and applying appropriate techniques to handle it.

### 4. Data Type Conversion
We convert data columns to appropriate data types to optimize memory usage and prepare for feature engineering. This step ensures consistency across all columns.

### 5. Data Exploration
Initial exploratory data analysis (EDA) is performed to understand the dataset's structure and characteristics. Key features are visualized to gain insights into the data.

### 6. Feature Engineering
New features are created from the existing data to enhance model performance and capture additional insights. This includes transformations and derived features.

### 7. Dataset Splitting
The dataset is split into training and testing subsets to prepare for model development and evaluation. This step ensures reproducibility and robust performance metrics.

### 8. Outlier Detection
Outliers in the dataset are identified and addressed to ensure they do not negatively impact the analysis or models.

### 9. Clustering Customers
We identify the most common customer profiles via clustering to understand different customer segments better.

### 10. Model Evaluation
Finally, we evaluate the performance of our models using various metrics and save the evaluation results for further analysis.

## Model Predictino

### Models Analyzed

In the analysis phase, we explored several machine learning models to predict customer churn. Each model was evaluated based on its performance metrics, such as accuracy, precision, recall, and F1-score. The models analyzed include:

1. **Logistic Regression**
    - A baseline model to understand the relationship between features and the probability of churn.
    - Easy to interpret and implement.

2. **Decision Tree**
    - A non-linear model that captures complex interactions between features.
    - Provides a visual representation of decision rules.

3. **Random Forest**
    - An ensemble method that combines multiple decision trees to improve performance.
    - Reduces overfitting and increases accuracy.

4. **Support Vector Machine (SVM)**
    - A powerful classifier that finds the optimal hyperplane to separate classes.
    - Effective in high-dimensional spaces.

5. **K-Nearest Neighbors (KNN)**
    - A simple, instance-based learning algorithm.
    - Classifies a sample based on the majority class of its nearest neighbors.

6. **Gradient Boosting Machine (GBM)**
    - An ensemble technique that builds models sequentially to correct errors of previous models.
    - Known for high predictive accuracy.

7. **XGBoost**
    - An optimized implementation of gradient boosting.
    - Provides parallel tree boosting and is highly efficient.

8. **Neural Networks**
    - Deep learning models that capture complex patterns in the data.
    - Consists of multiple layers to learn hierarchical representations.

Each model was tuned and validated using cross-validation techniques to ensure robust performance. The results were compared to select the best-performing model for predicting customer churn.


## Model Evaluation and Findings

### Evaluation Metrics

To evaluate the performance of each model, we used the following metrics:
- **Accuracy**: The proportion of correctly classified instances.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positives.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

### Best Model

After evaluating all the models, **XGBoost** stood out as the best performer. This model consistently achieved the highest scores across various metrics such as accuracy, precision, recall, and F1-score. The reasons for its superior performance include its advanced implementation of gradient boosting, which effectively combines the predictions of multiple weak learners, and its ability to handle large datasets efficiently. These characteristics make XGBoost particularly well-suited for predicting customer churn in this project.


## Dashboard

The dashboard provides an interactive visualization of the analysis results. It allows users to explore key metrics, such as churn rates and customer segments, through various charts and graphs. Users can filter data, view model performance, and gain insights into factors influencing customer churn.