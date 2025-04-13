"""
Use Anirudh's data loader for Image Imputation Dataset generation.
"""
from main_dataloader import MultiModalDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
import os
from datetime import datetime
import time


def load_statistics(s1_path='s1_statistics.csv', s2_path='s2_statistics.csv', modis_path='modis_statistics.csv'):
    try:

        s1_stats = pd.read_csv(s1_path)
        s2_stats = pd.read_csv(s2_path)
        modis_stats = pd.read_csv(modis_path)

    except FileNotFoundError as e:

        print(f"Error: Could not find one of the CSV files: {e}")
        return None, None, None

    # Convert to dictionaries for easy lookup
    s1_min_max = {row['Band']: (row['Min'], row['Max'])
                  for _, row in s1_stats.iterrows()}
    s2_min_max = {row['Band']: (row['Min'], row['Max'])
                  for _, row in s2_stats.iterrows()}
    modis_min_max = {row['Band']: (row['Min'], row['Max'])
                     for _, row in modis_stats.iterrows()}

    return s1_min_max, s2_min_max, modis_min_max


def normalize_channel(channel, band_idx, modality_name, s1_stats_path='s1_statistics.csv', s2_stats_path='s2_statistics.csv', modis_stats_path='modis_statistics.csv'):
    s1_min_max, s2_min_max, modis_min_max = load_statistics(
        s1_stats_path, s2_stats_path, modis_stats_path)
    if modality_name == 'Sentinel-1':
        band_key = list(s1_min_max.keys())[band_idx] if band_idx < len(
            s1_min_max) else str(band_idx)
        vmin, vmax = s1_min_max.get(
            band_key, (channel.min(), channel.max()))
    elif modality_name == 'Sentinel-2':
        band_key = list(s2_min_max.keys())[band_idx] if band_idx < len(
            s2_min_max) else str(band_idx)
        vmin, vmax = s2_min_max.get(
            band_key, (channel.min(), channel.max()))
    elif modality_name == 'MODIS':
        band_key = list(modis_min_max.keys())[band_idx] if band_idx < len(
            modis_min_max) else str(band_idx)
        vmin, vmax = modis_min_max.get(
            band_key, (channel.min(), channel.max()))
    else:
        # For other modalities, use array min/max as fallback
        vmin, vmax = channel.min(), channel.max()

    channel = (channel - vmin) / \
        (vmax - vmin) if vmax != vmin else channel - vmin
    channel = np.clip(channel, 0, 1)
    return channel


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define the paths to your data directories
    sentinel1_dir = 'path/to/sentinel1'
    sentinel2_dir = 'path/to/sentinel2'
    modis_dir = 'path/to/modis'

    start_time = time.time()

    # Define your dataset
    dataset = MultiModalDataset(
        sentinel1_dir='/work/mech-ai-scratch/rtali/gis-sentinel1/final_s1',
        sentinel2_dir='/work/mech-ai-scratch/rtali/gis-sentinel2/final_s2_v3',
        modis_dir='/work/mech-ai-scratch/rtali/gis-modis/modis',
        # Adjust mean/std per channel if needed
        transform=transforms.Normalize(mean=[0.5], std=[0.5])
    )

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    dload_time = time.time() - start_time
    print(f"Data loading time: {dload_time:.2f} seconds")

    # Collect 5000 samples
    sampled_data = []
    for i, sample in enumerate(loader):
        sampled_data.append(sample)
        if i == 4999:
            break

    # Sampling Time
    sampling_time = time.time() - dload_time
    print(f"Sampling time: {sampling_time:.2f} seconds")

    """
    Convert to tensors
    """

    # Init tensor of Zeros
    tensor_size = 64
    u_net = np.zeros((5000, 7, tensor_size, tensor_size))

    # Fill the tensor with the data

    # Extract 2 bands of Sentinel-1
    for i, sample in enumerate(sampled_data):
        # Extract Sentinel-1 bands
        s1 = []
        s2 = []
        modis = []

        for band in sample.shape[0]:
            s1.append(normalize_channel(sample[band].numpy()))

        for band in sample.shape[1]:
            s2.append(normalize_channel(sample[band].numpy()))

        for band in sample.shape[2]:
            modis.append(normalize_channel(sample[band].numpy()))

        # Stack the bands
        s1_bands = np.stack(s1, axis=0)  # (2, 64, 64)
        s2_bands = np.stack(s2, axis=0)  # (13, 64, 64)
        modis_bands = np.stack(modis, axis=0)  # (7, 64, 64)

        # Create a tensor by stacking on the first dimension
        combined = np.concatenate([s1_bands, s2_bands, modis_bands], axis=0)

        # Fill the u_net tensor
        u_net[i] = combined

    # Save the tensor to a file
    np.save('./models/u_net_data.npy', u_net)
    print("Data saved to ./models/u_net_data.npy")

    # Print the shape of the tensor
    print(f"Tensor shape: {u_net.shape}")

    # Print the time taken for the entire process
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
