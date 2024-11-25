# Telco Customer Churn

In this project of the IE500 Data Mining Course at the University of Mannheim, we are analysing the Telco Customer Churn dataset.

Website of the Data
https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148&id=i9710CF25EF75468D95FFFC7D57D45204&objRef=i9710CF25EF75468D95FFFC7D57D45204&action=run&format=HTML&cmPropStr=%7B%22id%22%3A%22i9710CF25EF75468D95FFFC7D57D45204%22%2C%22type%22%3A%22reportView%22%2C%22defaultName%22%3A%22DAT00148%22%2C%22permissions%22%3A%5B%22execute%22%2C%22read%22%2C%22traverse%22%5D%7D

notes:
not yet done:
5_data exploration
7_handling outliers

missing: total evaluation, hyperparameter tuning, performance on k fold cross validation
oversampling, undersampling, grid search, pca

questions:
- should we convert missing values to 'Not available' or set to e.g. 0? - no
- After one-hot-encoding we get the encoding into 'true' and 'false', should we encode it into binary 0 and 1 to fit the other representations? yes
- what to do with not binary created features in feature engineering? encode them into 0, 1 or leave them like that?
- Do 116 features need PCA? - no
- different splitting methods or just one? - one



0. reason in train test split that we have done preprocessing that is correct to do on train and dest combined (bc e.g. scaling applied later)
1.1 correlation matrix in preprocessing
1. Outlier run for all models + reasoning why maybe not included only non-churn? + evaluation (do worse models get better?)
2. section for models includes hyperparameter tuning (what we did and tried, what worked)
3. evaluation: f1 score only, reason why e.g. accuracy is misleading
4. report table for prediction included, reason in evaluation on all models, not only best. (why we get those results), best only in last line.
