import os

# Path to your main folder
base_path = r"C:\Users\russj\.cache\kagglehub\datasets\amerii\spacenet-7-multitemporal-urban-development\versions\1\SN7_buildings_train_sample"

# Walk through only the folders under SN7_buildings_train_sample
for root, dirs, files in os.walk(base_path):
    # Get relative path for cleaner printing
    rel_path = os.path.relpath(root, base_path)
    
    # Skip the root itself
    if rel_path == ".":
        continue
    
    # Print the folder structure
    print(rel_path)
    
    # Optional: print subfolders inside each folder
    for d in dirs:
        print(f"  └─ {d}")