from kagglehub import dataset_download
import os
import pandas as pd

dataset_name = "amerii/spacenet-7-multitemporal-urban-development"

# Download all files to a folder called "spacenet_data"
dataset_path = dataset_download(dataset_name, "spacenet_data")

# List all files in the folder
all_files = os.listdir(dataset_path)
print("Files downloaded:", all_files)

# Filter for CSVs (or other supported formats)
csv_files = [f for f in all_files if f.endswith(".csv")]

# Load all CSVs into a single DataFrame
df_list = [pd.read_csv(os.path.join(dataset_path, f)) for f in csv_files]
full_df = pd.concat(df_list, ignore_index=True)

print("First 5 rows of combined dataset:")
print(full_df.head())
