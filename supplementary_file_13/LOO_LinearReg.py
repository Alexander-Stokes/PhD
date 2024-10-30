import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('3in1ViscThinInteract13n2.csv')

X = df.drop('Visc', axis='columns').values
y = df.Visc.values

predictions = []

loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    precdcition_test = reg.predict(X_test)
    predictions.append(precdcition_test[0])

MSE = mean_squared_error(y, predictions)
print('MSE', MSE)

R2 = r2_score(y, predictions)
print('R2', R2)

# lines 12-26 can be summerised using cross_val_score to get an array of MSE values for each fold
mse_scores = -cross_val_score(reg, X, y, cv=loo, scoring='neg_mean_squared_error')
# Print the mean MSE across all folds
print('Mean squared error across all folds =', mse_scores.mean())