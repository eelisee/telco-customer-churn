# Analysis Overview: Machine Learning Models for Customer Churn Prediction

In this phase, we explore various machine learning algorithms to predict customer churn using the preprocessed dataset. Each notebook corresponds to a specific model, detailing its methodology, implementation, and evaluation.

## Algorithms and Their Applications

1. **Naive Bayes**  
   Naive Bayes predicts customer churn by treating features (e.g., customer behavior, demographics) as independent. The model calculates the probability of churn based on these features using Bayes' theorem.

2. **Logistic Regression**  
   Logistic Regression is a binary classification model often used for churn prediction. It estimates the likelihood of churn based on features like subscription length, number of complaints, and usage frequency.

3. **K-Nearest Neighbors (KNN)**  
   KNN predicts churn by identifying the most similar customers and leveraging their behavior to infer the likelihood of churn. The prediction is based on the majority behavior of the closest neighbors.

4. **Decision Trees**  
   Decision Trees classify churn by splitting data into branches based on decision rules (e.g., "If usage is below a threshold, predict churn"). The tree is constructed by recursively selecting features that best separate churners from non-churners.

5. **Random Forest**  
   Random Forest improves prediction accuracy by using an ensemble of decision trees. Each tree is trained on a random subset of data and features, and predictions are aggregated for a final output.

6. **Support Vector Machines (SVM)**  
   SVM predicts churn by identifying the optimal hyperplane that separates churners from non-churners. Kernels can be used to handle non-linear relationships within the data.

7. **Multilayer Perceptron (MLP)**  
   MLP, a type of artificial neural network, is suited for high-dimensional data. It captures complex, non-linear patterns to predict churn based on multiple layers of interconnected neurons.

8. **Artificial Neural Networks (ANN)**  
   ANNs model intricate relationships between features (e.g., demographics, usage patterns) and churn. They are effective at identifying non-linear dependencies that traditional models might overlook.

9. **XGBoost**  
   XGBoost, an advanced gradient boosting algorithm, is employed to enhance churn prediction. It uses a series of weak learners, optimizing for speed and performance, and effectively handles non-linear patterns and interactions.

## Analysis Workflow
Each algorithm is implemented and evaluated in a separate Jupyter Notebook:
- **1_Naive_Bayes.ipynb**
- **2_Logistic_Regression.ipynb**
- **3_KNN.ipynb**
- **4_Decision_Trees.ipynb**
- **5_Random_Forest.ipynb**
- **6_SVM.ipynb**
- **7_MLP.ipynb**
- **8_ANN.ipynb**
- **9_XGBoost.ipynb**

This structured analysis allows us to compare the strengths and weaknesses of various approaches and select the most suitable model for customer churn prediction.



We did some hyperparameter optimization. I. e.

For **K-Nearest Neighbors (KNN)**, we tested values of k ranging from 1 to 20. The final model was selected based on the value of k that yielded the highest accuracy.