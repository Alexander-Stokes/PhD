import os
import csv
import pandas as pd
from sklearn import linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Set the path to the directory containing the edited CSV files
csv_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_stiffness/csv_directory'

# Set the path to the folder where we'll save the fitted stress-strain plot pictures and results
output_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_stiffness/linear_fit'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the fitting function (Ax^2)
def fit_function(stress_points, strain_points):

    # Convert strain_points to a 2D array
    X = np.array(strain_points[:22]).reshape(-1, 1)

    # Fit a linear regression model
    reg = lm.LinearRegression()
    reg.fit(X, stress_points[:22])

    # Calculate fitted stress values using the linear regression model
    fitted_stress = reg.predict(X)

    gradient = reg.coef_[0] * 1000

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(stress_points[:22], fitted_stress))

    # Calculate R^2
    r2 = r2_score(stress_points[:22], fitted_stress)

    return fitted_stress, gradient, rmse, r2

def get_stress_strain_data(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    stress_points = [float(row[6]) * 1000 for row in rows[1:]]  # Convert MPa to kPa
    strain_points = [float(row[7]) * 100 for row in rows[1:]]  # Convert to percentage

    return strain_points, stress_points

def plot_stress_strain(filename, strain_points, stress_points, fitted_stress, output_directory):
    plt.figure()
    plt.plot(strain_points, stress_points, 'bo', label='Stress-Strain values', markersize=2)
    plt.plot(strain_points[:22], fitted_stress, 'r--', label='Fit (mx + c)')
    plt.xlabel('Strain [%]')
    plt.ylabel('Stress [kPa]')
    plt.title(f'Stress-Strain Curve with Fit for {filename}')
    plt.legend()
    plt.savefig(os.path.join(output_directory, f'{filename}_linear25.png'))
    plt.close()

# Create a new DataFrame to store the results
results_df = pd.DataFrame(columns=['Filename', 'Modulus [kPa]', 'RMSE', 'r2'])

# Loop through each CSV file in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        # Get stress and strain data using the module function
        strain_points, stress_points = get_stress_strain_data(os.path.join(csv_directory, filename))

        # Fit the curve using the fitting function
        try:
            # Calculate fitted stress values for the entire range of strain points
            fitted_stress, gradient, rmse, r2 = fit_function(stress_points, strain_points)

            # Save the fitted stress-strain plot with the curve
            plot_stress_strain(filename, strain_points, stress_points, fitted_stress, output_directory)

            # Add the results to the DataFrame
            results_df = results_df.append({'Filename': filename,  'Modulus [kPa]': gradient, 'RMSE': rmse, 'r2': r2}, ignore_index=True)
        except:
            A_value = None

# Save the A values to a CSV file
results_file_path = os.path.join(output_directory, 'fit_results.csv')
results_df.to_csv(results_file_path, index=False)

print("Fitted stress-strain plots and results saved in the output directory.")
