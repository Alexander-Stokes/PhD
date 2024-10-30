import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# Initialise data frame
df = pd.read_csv('Conc_Ratios.csv')

#intration term to multipuly
#multiply = df.GEL * df.EDC * df.NHS
#df['Multiply'] = multiply

#interation term addition
#add = df.GEL + df.EDC + df.NHS
#df['Add'] = add

#interation term mean - using pd mean for the X columns only
XColMean = df.drop('Visc', axis='columns').mean(axis='columns').values

#Xi = df.Add.values
X = np.reshape(XColMean, (-1,1))
# X = df.drop('Visc', axis='columns').values
# double brackets indicate this is a data frame, and must be an object.
y = df.Visc

LRpreds = []
RFpreds = []
KRpreds = []
# with LOO we make n number of models (reg)
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Linear regression
    linear_reg = linear_model.LinearRegression()
    linear_reg.fit(X_train, y_train)
    prediction_test1 = linear_reg.predict(X_test)
    LRpreds.append(prediction_test1[0])
    # Random Forest
    RF_reg = RandomForestRegressor()
    RF_reg.fit(X_train, y_train)
    prediction_test2 = RF_reg.predict(X_test)
    RFpreds.append(prediction_test2[0])
    # Kernel Ridge Classifier
    Kernel_reg = kernel_ridge.KernelRidge()
    Kernel_reg.fit(X_train, y_train)
    prediction_test3 = Kernel_reg.predict(X_test)
    KRpreds.append(prediction_test3[0])


print('Linear Regression Classifier')
print('MSE =', mean_squared_error(y,LRpreds))
print('r2 =', r2_score(y,LRpreds))
print('Random Forrest Classifier')
print('MSE =', mean_squared_error(y,RFpreds))
print('r2 =', r2_score(y,RFpreds))
print('Kernel Ridge Classifier')
print('MSE =', mean_squared_error(y,KRpreds))
print('r2 =', r2_score(y,KRpreds))

