import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('3in1ViscThinInteract13n2.csv')

X = df.drop('Visc', axis='columns').values
y = df.Visc.values

loo = LeaveOneOut()

# Set up the Ridge regression model
ridge = Ridge()

# Define the hyperparameter grid
param_grid = {'alpha': np.logspace(np.log10(0.1), np.log10(100000000), num=1000)}

# Set up Leave-One-Out cross-validation
loo = LeaveOneOut()

# Define scoring metrics (MSE and R2)
scoring = {'MSE': 'neg_mean_squared_error'}

# Perform GridSearchCV
grid_search = GridSearchCV(ridge, param_grid, cv=loo, scoring=scoring, refit='MSE')
grid_search.fit(X, y)

# Get the results
results = pd.DataFrame(grid_search.cv_results_)

# Print the best hyperparameters and corresponding MSE and R2
best_alpha = grid_search.best_params_['alpha']
best_mse = -grid_search.best_score_


print(f'Best alpha: {best_alpha}')
print(f'Best mean squared error: {best_mse}')

