import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import numpy as np

df = pd.read_csv('3in1ViscThinInteract13n2.csv')


X = df.drop('Visc', axis='columns').values
y = df.Visc.values

loo = LeaveOneOut()

# Set the alpha values for Ridge regression
alpha_values = np.linspace(0.1, 100000000.0, num=10000)  # You can add more values to the list
# Create a new pd dataframe to store the results
results_df = pd.DataFrame(columns=['Alpha', 'R2'])

for alpha in alpha_values:
    predictions = []
    r2_scores = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train)
        prediction = reg.predict(X_test)
        predictions.append(prediction[0])

    r2_scores.append(r2_score(y, predictions))
    r2_mean = sum(r2_scores) / len(r2_scores)

    new_row = pd.DataFrame({'Alpha': [alpha], 'R2': [r2_mean]})
    # Concatenate the existing DataFrame and the new row
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    # results_df = results_df.append({'Alpha': alpha, 'R2': r2_mean}, ignore_index=True)

results_df.to_csv('alphaVals.csv', index=False)
print('Finito!')

