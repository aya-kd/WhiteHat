import os
import pandas as pd

def count_rows_in_csv_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter only CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    
    # Iterate over each CSV file
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Print the number of rows in the DataFrame
        print(f"Number of rows in '{file}': {len(df)}")

# Example usage:
folder_path = './Dataset/CSV/Total'
count_rows_in_csv_files(folder_path)
