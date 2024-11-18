# Telco Customer Churn

In this project of the IE500 Data Mining Course at the University of Mannheim, we are analysing the Telco Customer Churn dataset.

Website of the Data
https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148&id=i9710CF25EF75468D95FFFC7D57D45204&objRef=i9710CF25EF75468D95FFFC7D57D45204&action=run&format=HTML&cmPropStr=%7B%22id%22%3A%22i9710CF25EF75468D95FFFC7D57D45204%22%2C%22type%22%3A%22reportView%22%2C%22defaultName%22%3A%22DAT00148%22%2C%22permissions%22%3A%5B%22execute%22%2C%22read%22%2C%22traverse%22%5D%7D

notes:
not yet done:
5_data exploration
7_handling outliers

missing: total evaluation, hyperparameter tuning, performance on k fold cross validation

questions:
- should we convert missing values to 'Not available' or set to e.g. 0?
- After one-hot-encoding we get the encoding into 'true' and 'false', should we encode it into binary 0 and 1 to fit the other representations?
- what to do with not binary created features in feature engineering? encode them into 0, 1 or leave them like that?
- Do 116 features need PCA?
- different splitting methods or just one?