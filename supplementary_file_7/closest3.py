import csv
import numpy as np

# Read the CSV file
csv_file = "LbL contact angle.csv"

# Initialize lists to store data for each row
output_data = []

# Read and process the CSV file
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if i == 0:
            continue  # Skip header row

        # Convert row values to floats (ignore the first column)
        data = [float(value) if value != '' else float('nan') for value in row[1:]]

        # Find the three nearest values
        nearest_indices = np.argsort(np.abs(np.array(data) - np.nanmean(data)))[:3]
        nearest_values = [data[index] for index in nearest_indices]

        # Calculate mean and standard deviation
        mean = np.nanmean(nearest_values)
        std_dev = np.nanstd(nearest_values)

        # Prepare data for output
        output_row = [f"Sample {i + 1}"] + nearest_values + [mean, std_dev]
        output_data.append(output_row)

# Write the data to the new CSV file
output_csv_file = "results_LBL.csv"
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sample", "Nearest Value 1", "Nearest Value 2", "Nearest Value 3", "Mean", "Standard Deviation"])
    writer.writerows(output_data)

print(f"Data saved to {output_csv_file}")
