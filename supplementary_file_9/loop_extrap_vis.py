import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import csv

# Set the path to the directory containing the edited CSV files
csv_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_viscosity/separated_samples'

# Define the directory name for saving graphs
output_directory = "Vis_Extrap"

# Create the "graphs" directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the modified power-law function
def modified_power_law(x, A, K, n):
    return (A + (K * (x ** (n-1))))

# Load the empirical shear constants from the CSV file
empirical_shear_constants_file = 'EmpiricalShearConstants.csv'
with open(empirical_shear_constants_file, 'r') as file:
    reader = csv.reader(file)
    # Skip the first row (header)
    next(reader)
    extrap_shear_values = [float(row[2]) for row in reader]

# Loop through each CSV file in the directory
for i, filename in enumerate(os.listdir(csv_directory)):
    if filename.endswith('.csv'):
        # Open the CSV file for reading
        with open(os.path.join(csv_directory, filename), 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)

        # Del the .1 in the filename.
        title = filename[:-6]

        extrap_shear = extrap_shear_values[i]  # Use the extrap_shear for this specific file

        # Extract data for plotting
        shear_rate = [float(row[0]) for row in rows[1:]]  # Column 0 for x-axis
        Avg_viscosity = [float(row[4]) for row in rows[1:]]  # Column 4 for y-axis
        sd = [float(row[5]) for row in rows[1:]]  # Column 2 for y-axis

        # Perform the fitting for combined data
        params, _ = curve_fit(modified_power_law, shear_rate, Avg_viscosity)
        # Extract A, K, and (n-1) values from the fitting parameters
        A, K, n = params

        # Calculate the fitted values using the modified power-law function
        fitted_viscosity = modified_power_law(shear_rate, A, K, n)
        extrap_viscosity = modified_power_law(extrap_shear, A, K, n)

        # Plot the combined data and the fitted curve
        plt.semilogx(shear_rate, Avg_viscosity, 'bo', label='Average Value')
        plt.errorbar(shear_rate, Avg_viscosity, yerr=sd, fmt='none', color='b')
        plt.semilogx(shear_rate, fitted_viscosity, 'r-', label='Power Law Fluid Model')
        # Plot the extrapolated vertical line
        plt.axvline(x=extrap_shear, color='r', linestyle='--')
        # Plot the extrapolated horizontal line
        plt.axhline(y=extrap_viscosity, color='r', linestyle='--', label='Extrapolated Viscosity')

        # Add labels and title
        plt.xlabel('Shear Rate [s-1]')  # Replace with appropriate x-axis label
        plt.ylabel('Viscosity [Pas]')  # Replace with appropriate y-axis label
        plt.title(f'Flow curve of Sample {title}')
        plt.legend()

        # Save the graph as an image with a filename based on the column name
        plt.savefig(os.path.join(output_directory, f'{title}_graph.png'))

        # Show the plot
        plt.close()

print("Flow curve plots with the power law fit saved in the output directory.")
