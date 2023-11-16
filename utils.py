import mne
import os
import glob
import pandas as pd
import numpy as np
import scipy.signal as signal


def gdf_to_df(file_path):
    raw_gdf = mne.io.read_raw_gdf(file_path)  # Use the full path here
    df = raw_gdf.to_data_frame()
    return df



def gdf_to_csv(file_path):
    raw_gdf = mne.io.read_raw_gdf(file_path)  # Use the full path here
    df = raw_gdf.to_data_frame()
    file_name = file_path.replace('.gdf', '.csv')
    df.to_csv(f'csv/{file_name}', sep=',', encoding='utf-8')

def csv_to_df(file_path):
    giga_df = pd.DataFrame()

    csv_files = glob.glob(file_path)
    print("Creating dataframe...")
    for file in csv_files:
        df = pd.read_csv(file, index_col=False, dtype=float)
        giga_df = pd.concat([giga_df, df], ignore_index=True)
    
    return giga_df

def format_df(df):
    print("Formatting dataframe...")
    df = df.drop(df.columns[0], axis=1)
    df = pd.DataFrame(df.values)
    return df

def band_pass_filter(data_arr,sampling_rate, cutoff,band_type):
    print("Applying " + band_type + " pass filter...")
    length = len(data_arr)
    # nyquist frequency = sampling / 2
    nyq = 0.5 * sampling_rate
    # butterworth filter requires normalized values
    normalized_cutoff = cutoff / nyq
    b, a = signal.butter(1, normalized_cutoff, btype=band_type, analog=False)
    result_arr = np.empty_like(data_arr)
    for i in range(length):
        result_arr[i] = signal.filtfilt(b, a, data_arr[i], axis=0)
    print(band_type + " pass filter applied.")
    return result_arr

def pad_arrays(array):
    max_length = max(data.shape[1] for data in array)
    array = [np.pad(data, ((0, 0), (0, max_length - data.shape[1])), mode='constant') for data in array]
    return np.array(array)