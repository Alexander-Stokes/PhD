import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Load the CSV file
csv_directory = 'C:/Users/alexs/PycharmProjects/Hydrogel/hydrogel_time/gelled_before'

# Define the directory name for saving graphs
output_directory = "1.5minsTime0predictionData"

# Create the "graphs" directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Loop through each CSV file in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(file_path)

        # Select the first 10 values for linear regression training
        X = df['Test Time'][:5].values.reshape(-1, 1)
        y_storage = df['Storage Modulus'][:5].values
        y_loss = df['Loss Modulus'][:5].values

        # Fit linear regression models
        model_storage = LinearRegression().fit(X, y_storage)
        model_loss = LinearRegression().fit(X, y_loss)

        # Generate 11 equally spaced values between 0 and the first real value in 'Test Time'
        time_predictions = np.linspace((df['Test Time'].iloc[0] - 90), df['Test Time'].iloc[0], 41).reshape(-1, 1)
        # delet the last value in (5,1) array because it is the same as the 1st real val.
        time_predictions = np.delete(time_predictions, -1, axis=0)

        # Predict 'Storage Modulus' and 'Loss Modulus' for the generated time values
        storage_predictions = model_storage.predict(time_predictions)
        loss_predictions = model_loss.predict(time_predictions)

        # Create a DataFrame with the predictions
        predictions_df = pd.DataFrame({
            'Test Time': time_predictions.flatten(),
            'Storage Modulus': storage_predictions,
            'Loss Modulus': loss_predictions
        })

        # Concatenate the original DataFrame with the predictions and save to a new CSV file
        output_df = pd.concat([predictions_df, df], ignore_index=True)
        output_file_path = os.path.join(output_directory, f'{filename}')
        output_df.to_csv(output_file_path, index=False)

print('New csv files are done!')
