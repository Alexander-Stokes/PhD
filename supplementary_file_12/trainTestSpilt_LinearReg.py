import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
import os

output_directory = "Graphs"

# Create the "graphs" directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialise data frame
df = pd.read_csv('../DATAcsv/3in1ViscNewtNorm.csv')
# dropping the visc (dependent variable) form the data frame.
# x_df includes GEL, Xconc, EDC
X = df.drop(['Visc', 'Hydrogel'], axis='columns')
# double brackets indicate this is a data frame, and must be an object.
y = df.Visc

title = 'Test Data Standardised = 20%'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
reg = linear_model.LinearRegression()
# fit take to arguments X (3 independent variables (GEL, EDC, NHS)) and Y (1 dependent variables (Visc))
reg.fit(X_train, y_train)

y_prediction = reg.predict(X_test)

# creating a list of the hydrogels take for testing
Hydro = []
for val in y_test.index:
    H = val + 1
    Hydro.append(H)

yTestList = y_test.tolist()

# Calculate Mean Squared Error (MSE)
MSE = mean_squared_error(y_test, y_prediction)
# Calculate R-squared (R²)
r2 = r2_score(y_test, y_prediction)

plt.plot(Hydro, yTestList, 'bo', label='Actual')
plt.plot(Hydro, y_prediction, 'ro', label='Predicted')
plt.xlabel('Hydrogels')  # Replace with appropriate x-axis label
plt.ylabel('Viscosity [mPa]')  # Replace with appropriate y-axis label
plt.title(f'Real vs Predicted - {title}')
# Set x-axis ticks to only the values in Hydro
plt.xticks(Hydro)
plt.legend()

# Open the CSV file in append mode
with open('metrics.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    # Check if the file is empty, if yes, write the header
    if file.tell() == 0:
        writer.writerow(['Test', 'MSE', 'r2'])

    # Write the values to the CSV file
    writer.writerow([title, MSE, r2])

plt.savefig(os.path.join(output_directory, f'{title}_graph.png'))
print('MSE and R2 are saved to the csv as well as the graph.')

