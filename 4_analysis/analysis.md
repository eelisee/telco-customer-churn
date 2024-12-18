# Analysis Overview: Machine Learning Models for Customer Churn Prediction

In this phase, we explore various machine learning algorithms to predict customer churn using the preprocessed dataset. Each notebook corresponds to a specific model, detailing its methodology, implementation, and evaluation.

## Algorithms and Their Applications

1. **Gaussian Naive Bayes**  
   Gaussian Naive Bayes predicts customer churn by treating features (e.g., customer behavior, demographics) as independent. The model calculates the probability of churn based on these features using Bayes' theorem and implying that all features are normally distributed.

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

(8. **Artificial Neural Networks (ANN)**)
   ANNs model intricate relationships between features (e.g., demographics, usage patterns) and churn. They are effective at identifying non-linear dependencies that traditional models might overlook. In our Project Proposal we have stated, that we wanted to implement this method but since ANN is essentially a logistic regression, we will not be implementing it here again. Instead we decided on implementing XGBoost.

Additionally, to the methods stated in our project proposal, we are also going to try out XGBoost, Nearest Centroid and Multinomial Naive Bayes. We will evaluate Multinomial Naive Bayes by itself as well as a combined version of both Multinomial and Gaussian Naive Bayes.

8. **XGBoost**  
   XGBoost, an advanced gradient boosting algorithm, is employed to enhance churn prediction. It uses a series of weak learners, optimizing for speed and performance, and effectively handles non-linear patterns and interactions.
   
9. **Nearest Centroid**  
   Nearest Centroid classifies churn by calculating the centroid (mean) of each class in the feature space and assigning new instances to the class with the nearest centroid. This method is simple and effective for datasets where classes are well-separated.

10. **Multinomial Naive Bayes**  
      Multinomial Naive Bayes is particularly suited for discrete data, such as word counts in text classification. For churn prediction, it models the probability of different feature values occurring in each class, assuming features are conditionally independent given the class.


## Analysis Workflow
Each algorithm is implemented and evaluated in a separate Jupyter Notebook:
- **1_Gaussian_Naive_Bayes.ipynb**
- **2_Logistic_Regression.ipynb**
- **3_KNN.ipynb**
- **4_Decision_Trees.ipynb**
- **5_Random_Forest.ipynb**
- **6_SVM.ipynb**
- **7_MLP.ipynb**
- **8_XGBoost.ipynb**
- **9_Nearest_Centroid.ipynb**
- **10_Multinomial_Naive_Bayes**

This structured analysis allows us to compare the strengths and weaknesses of various approaches and select the most suitable model for customer churn prediction.

We did some hyperparameter optimization. I. e.

For **K-Nearest Neighbors (KNN)**, we tested values of k ranging from 1 to 20. The final model was selected based on the value of k that yielded the highest accuracy.

And for **GNB**, we used GridSearch to optimize the hyperparamter. We also used GridSearch for **MNB** hyperparameter optimization.

For **Support Vector Machines (SVM)**, we used a pipeline with `StandardScaler` and `SVC`, and performed hyperparameter optimization using `RandomizedSearchCV` with a search space for `C`, `kernel`, and `gamma` parameters.

We applied cross-validation to each method to ensure the robustness and generalizability of our models. By partitioning the dataset into multiple folds, we trained and validated each model on different subsets of the data, thereby reducing the risk of overfitting and providing a more reliable estimate of model performance.

For each algorithm, we saved two output files: one for the cross-validation results on the training set and one for the predictions on the unseen data. Both files contain scores for accuracy, precision, recall, F1, and ROC AUC. This approach ensures that we have a comprehensive evaluation of each model's performance on both the training and test datasets.