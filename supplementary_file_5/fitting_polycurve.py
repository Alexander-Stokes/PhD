import os
import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

csvfile = 'rep26.1Data.csv'
title = 'Sample 26.1'
output_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_stiffness/best_fit1graph'

# Define the polynomial fitting function
def fit_function(x, k, n):
    return k*x**n

def plot_stress_strain(filename, strain_points, stress_points, fitted_stress):
    plt.figure()
    plt.plot(strain_points, stress_points, 'bo', markersize=2, label='Stress-Strain Curve')
    plt.plot(strain_points, fitted_stress, 'r--', label='Fit (kx^n)')
    plt.xlabel('Strain [%]')
    plt.ylabel('Stress [kPa]')
    plt.title(f'Stress-Strain Curve with Fit for {filename}')
    plt.legend()
    plt.savefig(os.path.join(output_directory, f'{filename}_poly_fit.png'))
    plt.close()

def get_stress_strain_data(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    stress_points = [float(row[6]) * 1000 for row in rows[1:]]  # Convert MPa to kPa
    strain_points = [float(row[7]) * 100 for row in rows[1:]]  # Convert to percentage

    return strain_points, stress_points

# Get stress and strain data using the module function
strain_points, stress_points = get_stress_strain_data(csvfile)
print(strain_points)
print(stress_points)

popt, _ = curve_fit(fit_function, strain_points, stress_points)
k_value, n_value = popt

print(popt)

# Calculate fitted stress values using the polynomial curve
fitted_stress = []
for value in strain_points:
    fitted_per_point = fit_function(value, k_value, n_value)
    fitted_stress.append(fitted_per_point)

print(fitted_stress)

plot_stress_strain(title, strain_points, stress_points, fitted_stress)
print(k_value, n_value)








