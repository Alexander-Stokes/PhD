import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, GridSearchCV

# Load data
df = pd.read_csv("Training Data/StiffTrainingData.csv")
folds = pd.read_csv("Testing Data/fakeProtoOutOfRangeInteract.csv")

# Features and target
X = df.drop('Stiff', axis='columns')
X = X.drop(['GEL', 'EDC', 'NHS', 'd', 'e'], axis='columns').values
y = df.Stiff.values

# Cross-validation setup
loo = LeaveOneOut()

# Hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Random Forest Regressor
RF_reg = RandomForestRegressor()

# Grid Search
grid_search = GridSearchCV(estimator=RF_reg, param_grid=param_grid, cv=loo, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Training with the best hyperparameters
best_RF_reg = grid_search.best_estimator_
