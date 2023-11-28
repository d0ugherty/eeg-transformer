import mne
import os
import glob
import pandas as pd
import numpy as np
import torch
import random
import scipy.signal as signal

"""
    Simple file conversion and formatting functions that may
    or may not be needed
"""

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

"""
    Cleans the data by applying band-pass filtering 
    dropping bad epochs
"""

def band_pass_filter(eeg_files):
    raw_objects = []
    epoch_objects = []
    for file in eeg_files:
        raw = mne.io.read_raw_gdf(file, preload=True)
        print(file)
        events, event_id = mne.events_from_annotations(raw)
        print(f'EVENT ID: {event_id}')
        tmin, tmax = -0.3, 0.7
        eeg_epochs = mne.Epochs(raw, events,event_id, event_repeated='merge', tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True)
        raw = raw.filter(l_freq=.1, h_freq=40.0, verbose=False)
        epoch_objects.append(eeg_epochs)
        raw_objects.append(raw)
    return raw_objects, epoch_objects

"""
    Converts the raw data into 
    PyTorch tensors
"""

def raw_to_tensor(raw_files):
    eeg_arrays = []

    for raw in  raw_files:
        data = raw.get_data()
        #print(f'From Raw: {data}')
        eeg_arrays.append(data)

    eeg_arrays = pad_arrays(eeg_arrays)
    eeg_data = torch.Tensor(eeg_arrays)
    return eeg_data
    

def pad_arrays(array):
    max_length = max(data.shape[1] for data in array)
    array = [np.pad(data, ((0, 0), (0, max_length - data.shape[1])), mode='constant') for data in array]
    return np.array(array)

"""
    Splits the data into training and validation sets
"""
def split_data(eeg_data, file_path, ratio):
    file_tensors = []
    start = 0
    for length in get_lengths(file_path):
        end = start + length
        file_tensor = eeg_data[start:end, :]
        
        file_tensors.append(file_tensor)
        start = 0


    random.shuffle(file_tensors)
    split_index = int(len(file_tensors) * ratio)
    train_files = file_tensors[:split_index]
    val_files = file_tensors[split_index:]
    
    return train_files, val_files

def get_lengths(file_path):
    file_lengths = []

    for file in file_path:
        raw = mne.io.read_raw_gdf(file, preload=True)
        length = len(raw.times)
        file_lengths.append(length)
    return file_lengths

def pad_tensors(batch):
    new_tensors = []
    max_size = 0
    for tensor in batch:
        if tensor.size(0) > max_size:
                max_size = tensor.size(0)
    
    for tensor in batch:
        if tensor.size(0) < max_size:
        
            padding = max_size - tensor.size(0)
            new_tensor = torch.nn.functional.pad(tensor, (padding,0,0,0))
            new_tensors.append(new_tensor)
        else:
            new_tensors.append(tensor)
    
    return torch.stack(new_tensors)

"""
    Splits a tensor into smaller tensors of a specified size along the second dimension.
"""
def split_tensor(tensor, split_size):
    num_splits = tensor.shape[1] // split_size
    splits = torch.split(tensor, split_size, dim=1)
    if len(splits) > num_splits:
        splits = splits[:num_splits]

    return splits
    
"""
    Shifts the EEG data in time.
"""
def time_shift(eeg_data, shift):
    return np.roll(eeg_data, shift, axis=1)

"""
    Adds random noise to the EEG data.
"""
def add_noise(eeg_data, noise_level):
    noise = np.random.normal(0, noise_level, eeg_data.shape)
    return eeg_data + noise

"""
    Warps the EEG data in time (speed up or slow down) for each channel.
"""
def time_warp(eeg_data, factor):
    if eeg_data.ndim != 2:
        raise ValueError("eeg_data should be a 2D array")
    num_channels, time_points = eeg_data.shape
    new_time_points = int(time_points * factor)
    warped_data = np.zeros((num_channels, new_time_points))

    for i in range(num_channels):
        warped_data[i, :] = np.interp(
            np.linspace(0, time_points, new_time_points),
            np.arange(time_points),
            eeg_data[i, :]
        )
    return warped_data

"""
    Apply a series of augmentations to a data.
"""
def augment_data(data, augmentations):
    augmented_data = data.copy()
    for augmentation in augmentations:
        augmented_data = augmentation(augmented_data)
    return augmented_data
