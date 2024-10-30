import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

# Initialise data frames
train_test_data = pd.read_csv("Training Data/TgelSynTestTrain.csv")
train_only_data = pd.read_csv("Training Data/TgelSynTrainOnly.csv")
folds = pd.read_csv("Testing Data/fakeProtoOutOfRangeInteract.csv")

# assign inputs and outputs keep separate for now
X = train_test_data.drop(['Tgel', 'd', 'e', 'f'], axis='columns').values
W = train_only_data.drop(['Tgel', 'd', 'e', 'f'], axis='columns').values
y = train_test_data['Tgel'].values
v = train_only_data['Tgel'].values
H = folds.drop(['d', 'e', 'f'], axis='columns').values

# List to store predictions for each iteration
RFpreds = []

# Leave-One-Out train-test split
loo = LeaveOneOut()

# Initialize an array to store predictions for each iteration
predictions = np.zeros(len(H))

# Training and testing within the LOOCV loop
for train_index, test_index in loo.split(X):
    X_train, X_test = np.vstack((X[train_index], W)), X[test_index]
    y_train = np.append(y[train_index], v)

    # Random Forest
    RF_reg = RandomForestRegressor()
    RF_reg.fit(X_train, y_train)

    # Testing
    for idx, i in enumerate(H):
        prediction = RF_reg.predict([i])
        predictions[idx] += prediction[0]

    No_splits = 0
    No_splits += 1

# Compute the average prediction across all iterations
average_predictions = predictions / No_splits

# Print or use the average predictions as needed
for pred in average_predictions:
    print(pred)
