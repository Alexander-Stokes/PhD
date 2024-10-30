import matplotlib.pyplot as plt
import os
import pandas as pd

# Set the path to the directory containing the edited CSV files
csv_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_time/1.5minsTime0predictionData'

# Define the directory name for saving graphs
output_directory = "loglogAxisCrossoverTimeSweepGraphsT0predic1.5mins"

# Create the "graphs" directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through each CSV file in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        # Open the CSV file for reading
        file_path = os.path.join(csv_directory, filename)
        data = pd.read_csv(file_path)

        # Del the .csv in the filename.
        title = filename[:-4]

        # Extract data for plotting
        time = data['Test Time']
        store = data['Storage Modulus']
        loss = data['Loss Modulus']

        time_minutes = [t / 60.0 for t in time]

        crossover = None

        # Iterate through the lists until 'store' becomes greater than 'loss'
        # if hydrogel has already gelled at time[0] don't assign a crossover val.
        if loss[0] > store[0]:
            for i in range(len(loss)):
                if store[i] > loss[i]:
                    crossover = i
                    break

        # 'crossover' now contains the index where 'store' becomes greater than 'loss'.
        if crossover is not None:
            # time sweep sampling frequency - 30s, minus 15s for the actual time.
            geltime = time_minutes[crossover]

        # Plot the loss and storage modulus with and x-y log scale
        # only plot the 1st 55 data points of the csv files
        plt.plot(time_minutes[:55], store[:55], 'b.', label='Storage Modulus')
        plt.plot(time_minutes[:55], loss[:55], 'r.', label='Loss Modulus')
        # here to stop the crossover line from appering when no gelation occours
        if crossover is not None:
            # mark the crossover point (geltime) with Vertial (axv) line
            plt.axvline(x=geltime, color='r', linestyle='--', label='Gelation time')

        # Add labels and title
        plt.xlabel('Time [minutes]')  # Replace with appropriate x-axis label
        plt.ylabel('Modulus [Pa]')  # Replace with appropriate y-axis label
        plt.title(f'Time sweep of Sample {title}')
        plt.legend()

        # Adding markers for 30s, 1m, 2m, 3m, 10m, 30m, 1h
        # x_ticks = [0.5, 1, 2, 3, 4, 5, 10, 30, 60]
        # plt.xticks(x_ticks, [str(x) for x in x_ticks])

        # Save the graph as an image with a filename based on the column name
        plt.savefig(os.path.join(output_directory, f'{title}_graph.png'))

        # Show the plot
        plt.close()

print("time sweep plots are now saved in the output directory.")
