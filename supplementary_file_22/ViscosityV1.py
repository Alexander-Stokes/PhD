import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('Training Data/ViscTrianingData.csv')
fold1000 = pd.read_csv('Testing Data/fakeProtoOutOfRangeInteract.csv')


X = df.drop('Visc', axis='columns').values
H = fold1000.values
y = df.Visc.values

loo = LeaveOneOut()

# Set the alpha values for Ridge regression
alpha = 8648423.275731727

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    print(prediction)
    print('done!')

# # Testing
# for i in H:
#     prediction = reg.predict([i])
#     print(prediction[0])