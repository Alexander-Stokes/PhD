import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn import kernel_ridge

# Initialise data frames
df = pd.read_csv("StiffTrainingData.csv")
folds = pd.read_csv("fakeProtoOutOfRangeInteract.csv")


# assign inputs and outputs keep separte for now
X = df.drop('Stiff', axis='columns')
# X Interaction terms X
# X = X.drop('GEL', axis='columns')
# X = X.drop('EDC', axis='columns')
# X = X.drop('NHS', axis='columns')
X = X.drop('d', axis='columns')
X = X.drop('e', axis='columns')
X = X.drop('f', axis='columns').values

y = df.Stiff

alpha = 536.6976945540476
preds = []

# abriviations
loo = LeaveOneOut()
alphaRidge = Ridge(alpha=alpha)
ridge = Ridge()
Kridge = kernel_ridge.KernelRidge()

# Training predicting
def main(reg):
    for train_index, test_index in loo.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Random Forest
        reg.fit(X_train, y_train)
        prediction_test = reg.predict(X_test)
        preds.append(prediction_test[0])

# Predictive cappacity
main(alphaRidge)
print('R2', r2_score(y,preds))
print(preds)
preds = []

main(ridge)
print('R2', r2_score(y,preds))
print(preds)
preds = []

main(Kridge)
print('R2', r2_score(y,preds))
print(preds)
preds = []

