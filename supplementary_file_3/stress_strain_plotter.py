import csv
import os
import matplotlib.pyplot as plt

# Set the path to the directory containing the edited CSV files
csv_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_stiffness/csv_directory'

# Set the path to the folder where we'll save the stress-strain plot pictures
output_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_stiffness/stress_strain_plots'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through each CSV file in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        # Open the CSV file for reading
        with open(os.path.join(csv_directory, filename), 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)

        # Extract the stress and strain data
        stress_points = [float(row[6]) * 1000 for row in rows[1:]]  # Convert MPa to kPa
        strain_points = [float(row[7]) * 100 for row in rows[1:]]   # Convert to percentage

        # Create the stress-strain plot
        plt.plot(strain_points, stress_points)
        plt.xlabel('Strain [%]')
        plt.ylabel('Stress [kPa]')
        plt.title(f'Stress-Strain Curve for {filename}')
        plt.savefig(os.path.join(output_directory, f'{filename}_stress_strain_plot.png'))
        plt.close()

print("Stress-strain plots saved in the output directory.")

