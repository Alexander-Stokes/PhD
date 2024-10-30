import pandas as pd
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut

# Create dataframe from dataset
df = pd.read_csv('3in1ViscThinInteract13n2.csv')

# Assign inputs and outputs
X = df.drop('Visc', axis='columns').values
y = df.Visc.values

loo = LeaveOneOut()

# Create a new pd dataframe to store the results
results_df = pd.DataFrame(columns=['Fold', 'Ridge', 'Linear'])
# Keep track of fold number
fold = 0

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Ridge regression
    rid = Ridge(alpha=8690869.17821782)
    rid.fit(X_train, y_train)
    rid_prediction = rid.predict(X_test)
    error1 = y_test - rid_prediction
    error1 = error1.tolist()
    error1 = error1[0]

    # Linear regression
    lin = linear_model.LinearRegression()
    lin.fit(X_train, y_train)
    lin_prediction = lin.predict(X_test)
    error2 = y_test - lin_prediction
    error2 = error2.tolist()
    error2 = error2[0]

    # Count fold
    fold += 1

    new_row = pd.DataFrame({'Fold': [fold], 'Ridge': [error1], 'Linear': [error2]})
    # Concatenate the existing DataFrame and the new row
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    # results_df = results_df.append({'Alpha': alpha, 'R2': r2_mean}, ignore_index=True)

results_df.to_csv('overfitting_ridge.csv', index=False)


