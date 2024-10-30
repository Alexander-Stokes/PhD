import csv
import os
import numpy as np

# Set the diameter constant to 10 mm
diameter = 10

# Set the path to the directory containing the edited CSV files
csv_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_stiffness/csv_directory'

# Create a new CSV file to write the Young's modulus values to
with open('E50.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Filename', 'Young\'s modulus'])

    # Loop through each CSV file in the directory
    for filename in os.listdir(csv_directory):
        if filename.endswith('.csv'):
            # Open the CSV file for reading
            with open(os.path.join(csv_directory, filename), 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)

                # Find the row with the closest strain value to 0.5 (50% strain)
                target_strain = 0.5
                closest_row = None
                closest_dist = float('inf')
                for row in rows:
                    try:
                        strain = float(row[7])
                        dist = abs(strain - target_strain)
                        if dist < closest_dist:
                            closest_row = row
                            closest_dist = dist
                    except ValueError:
                        pass

                # Calculate the Young's modulus using the formula E = stress / strain
                stress = float(closest_row[6])
                strain = float(closest_row[7])
                modulus = stress / strain

                # Write the filename and Young's modulus value to the output CSV file
                writer.writerow([filename, modulus])
