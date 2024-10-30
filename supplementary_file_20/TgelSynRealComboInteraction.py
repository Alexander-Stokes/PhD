import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Initialise data frames
train_test_data = pd.read_csv("TgelBOTHTrainTest.csv")
train_only_data = pd.read_csv("TgelBOTHTrainOnly.csv")

# assign inputs and outputs keep separte for now
X = train_test_data.drop('Tgel', axis='columns')
# X Interaction terms X
X = X.drop('GEL', axis='columns')
X = X.drop('EDC', axis='columns')
X = X.drop('NHS', axis='columns')
X = X.drop('d', axis='columns')
X = X.drop('e', axis='columns').values
# X = X.drop('f', axis='columns').values

W = train_only_data.drop('Tgel', axis='columns')
# W Interaction terms W
W = W.drop('GEL', axis='columns')
W = W.drop('EDC', axis='columns')
W = W.drop('NHS', axis='columns')
W = W.drop('d', axis='columns')
W = W.drop('e', axis='columns').values
# W = W.drop('f', axis='columns').values

y = train_test_data.Tgel
v = train_only_data.Tgel

# Regressors
linear_reg = linear_model.LinearRegression()
RF_reg = RandomForestRegressor()
Kernel_reg = kernel_ridge.KernelRidge()

# lists of predictions for each regressor.
y_predList = []
y_testList = []

# Initialize KFold
kf = KFold(n_splits=53)

# Iterate over each fold
def main(reg):
    """
    1. Splits the dataset using k-flod crossvalidation.
    2. Synthetic data is kept for training only
    3. Fits the model using reg
    4. Acesses the predictive capasity using MSE and R2
    5. Prints MSE then R2
    """
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Combining the LOO training set with the whole of the trainOnly dataset.
        X_train = np.vstack((X_train, W))
        # because this is a one dimensional array aka a list, we can use append.
        y_train = np.append(y_train, v)


        reg.fit(X_train, y_train)

        # Make predictions
        y_pred = reg.predict(X_test)
        global y_predList
        y_predList.append(y_pred)
        global y_testList
        y_testList.append(y_test.values)

    y_predList = np.array(y_predList)
    y_testList = np.array(y_testList)

    mse = mean_squared_error(y_testList, y_predList)
    r2 = r2_score(y_testList, y_predList)

    print(mse)
    print(r2)

def clear_lists():
    """to clear the list so the main function can run."""
    global y_predList
    global y_testList
    y_predList = []
    y_testList = []

# run with different regressors
main(linear_reg)
clear_lists()
main(RF_reg)
clear_lists()
main(Kernel_reg)