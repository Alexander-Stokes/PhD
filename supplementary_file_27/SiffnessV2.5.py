import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("Training Data/StiffTrainingData.csv") # 81 unrolled clossest 3 dataset
Hydrogels = pd.read_csv("Training Data/27XHydrogelsStiff.csv") # 27 rolled hydrogels
folds1000 = pd.read_csv("Testing Data/fakeProtoOutOfRangeInteract.csv") # X interaction terms for 1000 fake hydrogels

# inputs and
X = df.drop(['Stiff', 'GEL', 'EDC', 'NHS', 'd', 'e'], axis=1).values
H = Hydrogels.drop(['Stiff', 'GEL', 'EDC', 'NHS', 'd', 'e'], axis=1).values
F = folds1000.drop(['GEL', 'EDC', 'NHS', 'd', 'e'], axis=1).values
# outputs
y = df['Stiff'].values
Hy = df['Stiff'].values

# Define parameter grid for hyperparameter tuning
param_grid = {
    'bootstrap': [True],
    'max_depth': [1, 5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200]
}

# Perform nested cross-validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_scores = []
rowList = []
preds = []
Fold = 0
for train_index, test_index in outer_cv.split(X):
    Fold += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Inner cross-validation for hyperparameter tuning
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, scoring='r2', cv=inner_cv)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print('Best model', best_model)
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    prediction_test = best_model.predict(X_test)
    preds.append(prediction_test)
    print(f'predictions fold-{Fold}')
    print(prediction_test)
    print(f'Ground Truth fold-{Fold}')
    print(y_test)
    r2_score(y_test, prediction_test)
    outer_scores.append(r2_score(y_test,prediction_test))

avgR2 = sum(outer_scores)/len(outer_scores)
print(avgR2)

#     row27 = []
#
    for i in F:
        y_pred = best_model.predict([i])

        # row27.append(y_pred[0])
#
#     rowList.append(row27)
#
# crossVal_matrix = np.array(rowList)
# print(crossVal_matrix)
# column_averages = np.mean(crossVal_matrix, axis=0)
# print('Avg: ', column_averages)
#
# print(r2_score(y,column_averages))

#Evaluate the model on the test set
# for i in Hy:
#     outer_scores.append(r2_score(i, y_pred))

#Evaluate the model on the test set
# for i in H:
#     y_pred = best_model.predict([i])
#     outer_scores.append(r2_score(y_test, y_pred))
#
# # Calculate and print the average R^2 score
# avg_r2_score = np.mean(outer_scores)
# print("Average R^2 score:", avg_r2_score)
