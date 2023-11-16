import argparse
import os
import numpy as np
import pandas as pd
import mne
"""
    Read file into python
    Takes a directory as a command line argument and iterates over 
    each .gdf file to convert it into a python-readable format
"""
parser = argparse.ArgumentParser(description='Convert MATLAB .gdf files in a directory to a Python-readable format.')
parser.add_argument('dir_path', type=str, help='The path of the directory containing .gdf or .mat files to convert.')

args = parser.parse_args()
dir_path = args.dir_path

file_list = os.listdir(dir_path)

for file in file_list:
    if file.endswith('.gdf'):
        full_path = os.path.join(dir_path, file)  # Create the full path to the file
        print("File name: " + file)
            
        raw_gdf = mne.io.read_raw_gdf(full_path)  # Use the full path here
        df = raw_gdf.to_data_frame()
        file_name = file.replace('.gdf', '.csv')
        df.to_csv(f'csv/{file_name}', sep=',', encoding='utf-8')

