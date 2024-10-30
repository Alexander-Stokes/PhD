import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# Initialise data frames
train_test_data = pd.read_csv("synthetic compare/TgelSynTestTrain.csv")
train_only_data = pd.read_csv("synthetic compare/TgelSynTrainOnly.csv")
folds = pd.read_csv("synthetic compare/fakeProtoOutOfRangeInteract.csv")


# assign inputs and outputs keep separte for now
X = train_test_data.drop('Tgel', axis='columns')
# X Interaction terms X
# X = X.drop('GEL', axis='columns')
# X = X.drop('EDC', axis='columns')
# X = X.drop('NHS', axis='columns')
X = X.drop('d', axis='columns')
X = X.drop('e', axis='columns')
X = X.drop('f', axis='columns').values

W = train_only_data.drop('Tgel', axis='columns')
# W Interaction terms W
# W = W.drop('GEL', axis='columns')
# W = W.drop('EDC', axis='columns')
# W = W.drop('NHS', axis='columns')
W = W.drop('d', axis='columns')
W = W.drop('e', axis='columns')
W = W.drop('f', axis='columns').values

y = train_test_data.Tgel
v = train_only_data.Tgel

# Cross validation for each hydrogel dataframe setup
H = folds.drop('Tgel', axis='columns')
# H Interaction terms H
# H = H.drop('GEL', axis='columns')
# H = H.drop('EDC', axis='columns')
# H = H.drop('NHS', axis='columns')
H = H.drop('d', axis='columns')
H = H.drop('e', axis='columns')
H = H.drop('f', axis='columns').values


# lists of predictions for each regressor.
RFpreds = []
foldNum = 0

# Leave one out train test spilt
loo = LeaveOneOut()
# Training
for train_index, test_index in loo.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Combining the LOO training set with the whole of the trainOnly dataset.
    X_train = np.vstack((X_train, W))
    # because this is a one dimensional array aka a list, we can use append.
    y_train = np.append(y_train, v)

    # Random Forest
    RF_reg = RandomForestRegressor()
    RF_reg.fit(X_train, y_train)

# Testing
for i in H:
    prediction = RF_reg.predict([i])
    print(prediction[0])
    # RFpreds.append(prediction[0])




