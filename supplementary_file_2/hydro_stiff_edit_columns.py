import csv
import os
import numpy as np

# Set the diameter constant to 10 mm
diameter = 10

# Set the path to the directory containing the CSV files
KPa_csv_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_stiffness/kPa_csv_directory'

# Loop through each CSV file in the directory
for filename in os.listdir(KPa_csv_directory):
    if filename.endswith('.csv'):
        # Open the CSV file for reading and writing
        with open(os.path.join(KPa_csv_directory, filename), 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            # Check if the stress and strain columns already exist
            if 'stress' not in rows[0] and 'strain' not in rows[0]:
                # Add the column headers for stress and strain
                rows[0].extend(['stress', 'strain'])
                # Loop through each row of data in the CSV file
                for i in range(1, len(rows)):
                    # Calculate the initial height from the first value in column 5
                    initial_height = float(rows[1][3])
                    # Calculate stress using the formula force / (pi * diameter^2 / 4)
                    force = float(rows[i][5])
                    stress = (force / (np.pi * diameter**2 / 4)) * 1000 # to convert to KPa
                    # Calculate strain using the formula (initial height - current height) / initial height
                    strain = float(rows[i][4]) / initial_height
                    # Append the stress and strain values to the row
                    rows[i].extend([stress, strain])
                # Write the updated data back to the CSV file
                with open(os.path.join(KPa_csv_directory, filename), 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(rows)
