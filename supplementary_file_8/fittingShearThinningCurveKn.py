import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Define the modified power-law function
def modified_power_law(x, A, K, n):
    return (A + (K * (x ** (n-1))))

# Load the CSV file into a DataFrame
file_path = 'separated_samples/13.1.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Extract data for plotting
shear_rate = data['Shear Rate [1/s]']  # Column 0 for x-axis
Avg_viscosity = data['Avg']  # Column 4 for y-axis
sd = data['SD']  # Column 2 for y-axis

# Perform the fitting for combined data
params, _ = curve_fit(modified_power_law, shear_rate, Avg_viscosity)
# Extract K and (n-1) values from the fitting parameters
A, K, n = params

# Calculate the fitted values using the modified power-law function
fitted_viscosity = modified_power_law(shear_rate, A, K, n)

# Calculate Mean Squared Error (MSE)
MSE = mean_squared_error(Avg_viscosity, fitted_viscosity)

# Calculate R-squared (R²)
r2 = r2_score(Avg_viscosity, fitted_viscosity)

# Plot the combined data and the fitted curve
plt.semilogx(shear_rate, Avg_viscosity, 'bo', label='Average Value')
plt.errorbar(shear_rate, Avg_viscosity, yerr=sd, fmt='none', color='b')
plt.semilogx(shear_rate, fitted_viscosity, 'r-', label='Power Law Fluid Model')

# Add labels and title
plt.xlabel('Shear Rate [s-1]')  # Replace with appropriate x-axis label
plt.ylabel('Viscosity [mPas]')  # Replace with appropriate y-axis label
plt.title('Plot of Combined Data with Fitted Curve')
plt.legend()

extrap_viscosity = modified_power_law(0.449697371, A, K, n)



print('A:', A)
print('K:', K)
print('n:', n)
print('vis:', extrap_viscosity)
print('Mean Squared Error:', MSE)
print('R-squared:', r2)

# Show the plot
plt.show()