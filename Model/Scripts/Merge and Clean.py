################################################## Merge csv files and remove empty lines #####################################################""
import os
import pandas as pd

def merge_csv_files(folder_path, output_file, word_to_replace):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter only CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    
    # Initialize an empty DataFrame to store merged data
    merged_df = pd.DataFrame()
    
    # Iterate over each CSV file
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Change the values in the second column to the specified word
        df.iloc[:, 1] = word_to_replace
        # Remove rows where the first column is empty
        df = df[df.iloc[:, 0].notna()]
        # Concatenate the DataFrame with the merged DataFrame
        merged_df = pd.concat([merged_df, df], ignore_index=True, sort=False)
    
    # Write the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)

# Example usage:
folder_path = './Dataset/CSV/UC'
output_file = './Dataset/CSV/Total/UC.csv'
word_to_replace = 'UC'
merge_csv_files(folder_path, output_file, word_to_replace)
