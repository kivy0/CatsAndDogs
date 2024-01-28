import os
import random
from sklearn.model_selection import train_test_split
from shutil import move

def split_data(input_dir, output_train_dir, output_val_dir, test_size=0.2, random_seed=42):
    files = os.listdir(input_dir)
    train_files, val_files = train_test_split(files, test_size=test_size, random_state=random_seed)
    
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    
    for file in train_files:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(output_train_dir, file)
        move(src_path, dest_path)
    
    for file in val_files:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(output_val_dir, file)
        move(src_path, dest_path)

if __name__ == "__main__":
    input_data_dir = "data"
    output_train_data_dir = "train"
    output_val_data_dir = "val"
    test_split_size = 0.2
    random_seed = 42

    split_data(input_data_dir, output_train_data_dir, output_val_data_dir, test_size=test_split_size, random_seed=random_seed)

