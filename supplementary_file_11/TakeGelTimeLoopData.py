import os
import pandas as pd

# Set the path to the directory containing the edited CSV files
csv_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_time/NewStartTimes'


# Creating an empty DataFrame
blankTable = ['Sample', 'Tg']
listTg = pd.DataFrame(columns=blankTable)

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

        if crossover == None:
            lossLastVal = len(loss) - 1
            storeLastVal = len(store) - 1
            if store[0] > loss[0]:
                geltime = 0
            if loss[lossLastVal] > store[storeLastVal]:
                geltime = 60

        tableInput = {'Sample': f'{title}', 'Tg': geltime}
        new_df = pd.DataFrame([tableInput])

        listTg = pd.concat([listTg, new_df], ignore_index=True)


listTg.to_csv('GelationTimePredict1.5minsT0.csv', index=False)
print('Finito!')