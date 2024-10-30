import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

# Initialise data frames
df = pd.read_csv("Training Data/StiffTrainingData.csv")
folds = pd.read_csv("Testing Data/fakeProtoOutOfRangeInteract.csv")


# assign inputs and outputs keep separte for now
X = df.drop('Stiff', axis='columns')
# X Interaction terms X
X = X.drop('GEL', axis='columns')
X = X.drop('EDC', axis='columns')
X = X.drop('NHS', axis='columns')
X = X.drop('d', axis='columns')
X = X.drop('e', axis='columns').values
# X = X.drop('f', axis='columns')

y = df.Stiff


# Cross validation for each hydrogel dataframe setup
# H Interaction terms H
H = folds.drop('GEL', axis='columns')
H = H.drop('EDC', axis='columns')
H = H.drop('NHS', axis='columns')
H = H.drop('d', axis='columns')
H = H.drop('e', axis='columns').values
# H = H.drop('f', axis='columns')

preds = []
Hyperparameters = {'bootstrap': True, 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5,
                      'n_estimators': 50}
# Leave one out train test spilt
loo = LeaveOneOut()
# Training
for train_index, test_index in loo.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Random Forest
    RF_reg = RandomForestRegressor(**Hyperparameters)
    RF_reg.fit(X_train, y_train)
    prediction_test = RF_reg.predict(X_test)
    preds.append(prediction_test[0])
    print(prediction_test[0])

# Cross-validation
preds2 = []
for i in X:
    prediction = RF_reg.predict([i])
    preds2.append(prediction[0])

# Predictive capacity
print(preds)
print(preds2)
print(r2_score(y,preds))
print(r2_score(y,preds2))