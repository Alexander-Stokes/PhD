import csv
import os

# Set the path to the directory containing the CSV files
csv_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_stiffness/csv_directory'

# Loop through each CSV file in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        # Set the path to the CSV file
        csv_path = os.path.join(csv_directory, filename)
        # Open the CSV file for reading and writing
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            # Remove rows greater than 63
            rows = rows[:61]
        # Write the updated data back to the CSV file
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
