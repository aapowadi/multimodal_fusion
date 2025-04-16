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


def load_statistics(s1_path='s1_statistics.csv', s2_path='s2_statistics.csv', modis_path='modis_statistics.csv',
                    soil_path='soil_statistics.csv', weather_path='weather_statistics.csv'):
    try:
        s1_stats = pd.read_csv('statistics/' + s1_path)
        s2_stats = pd.read_csv('statistics/' + s2_path)
        modis_stats = pd.read_csv('statistics/' + modis_path)
        soil_stats = pd.read_csv('statistics/' + soil_path)
        weather_stats = pd.read_csv('statistics/' + weather_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the CSV files: {e}")
        return None, None, None, None, None

    # Convert to dictionaries for easy lookup
    s1_min_max = {row['Band']: (row['Min'], row['Max'])
                  for _, row in s1_stats.iterrows()}
    s2_min_max = {row['Band']: (row['Min'], row['Max'])
                  for _, row in s2_stats.iterrows()}
    modis_min_max = {row['Band']: (row['Min'], row['Max'])
                     for _, row in modis_stats.iterrows()}
    soil_min_max = {row['Band']: (row['Min'], row['Max'])
                    for _, row in soil_stats.iterrows()}
    weather_min_max = {row['Band']: (row['Min'], row['Max'])
                       for _, row in weather_stats.iterrows()}

    return s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max


def full_normalization(dataset, sample_idx=0):

    # Bring Normalization to the dataset from display_samples.py
    # Load statistics
    s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max = load_statistics()

    if s1_min_max is None:
        print("Using percentile-based normalization as fallback.")
        # Define fallback percentile-based normalization

        def normalize_channel(channel, percentile=2):
            vmin = np.percentile(channel, percentile)
            vmax = np.percentile(channel, 100 - percentile)
            channel = (channel - vmin) / \
                (vmax - vmin) if vmax != vmin else channel - vmin
            channel = np.clip(channel, 0, 1)
            return channel
    else:
        # Define min-max normalization using CSV statistics
        def normalize_channel(channel, band_idx, modality_name):
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
                # Remaining modality is CDL, which has min value = 0 and max value of 254
                vmin, vmax = 0.0, 254.0

            channel = (channel - vmin) / \
                (vmax - vmin) if vmax != vmin else channel - vmin
            channel = np.clip(channel, 0, 1)
            return channel

    # Normalize each channel
    modalities, label = dataset[sample_idx]
    modality_names = ['Sentinel-1', 'Sentinel-2', 'MODIS']

    for i, (modality, name) in enumerate(zip(modalities, modality_names)):

        if name == 'Sentinel-2':
            r = normalize_channel(
                modality[3].numpy(), band_idx=3, modality_name='Sentinel-2')
            g = normalize_channel(
                modality[2].numpy(), band_idx=2, modality_name='Sentinel-2')
            b = normalize_channel(
                modality[1].numpy(), band_idx=1, modality_name='Sentinel-2')
            rgb = np.stack([r, g, b], axis=-1)

        elif name == 'MODIS':

            r = normalize_channel(
                modality[0].numpy(), band_idx=0, modality_name='MODIS')
            g = normalize_channel(
                modality[3].numpy(), band_idx=3, modality_name='MODIS')
            b = normalize_channel(
                modality[2].numpy(), band_idx=2, modality_name='MODIS')
            rgb = np.stack([r, g, b], axis=-1)

        else:
            channel = normalize_channel(
                modality[0].numpy(), band_idx=0, modality_name=name)


if __name__ == "__main__":

    # Set random seed for reproducibility
    np.random.seed(42)

    start_time = time.time()

    '''
    We want to extract 5000 samples from the dataset. Prepare three numpy tensors. One for each modality.
    The first tensor will be for MODIS, the second for Sentinel-1, and the third for Sentinel-2.
    The dimensions of the tensors will be (5000, 7, 64, 64) for MODIS, (5000, 2, 64, 64) for Sentinel-1,
    and (5000, 12, 64, 64) for Sentinel-2.
    The first dimension is the number of samples, the second dimension is the number of channels,
    and the last two dimensions are the height and width of the image.
    The images will be 64x64 pixels in size.
    The channels for MODIS are 7, for Sentinel-1 are 2, and for Sentinel-2 are 12.
    '''

    NSAMPLES = 5000
    DIMX = 64
    DIMY = 64
    MODIS_DIMS = 7
    SENTINEL2_DIMS = 12
    SENTINEL1_DIMS = 2

    # Initialize a numpy tensor to store the samples. float32 for full precision
    unetX = np.zeros((NSAMPLES, MODIS_DIMS, DIMX, DIMY), dtype=np.float32)

    unetYS1 = np.zeros(
        (NSAMPLES, SENTINEL1_DIMS, DIMX, DIMY), dtype=np.float32)

    unetYS2 = np.zeros(
        (NSAMPLES, SENTINEL2_DIMS, DIMX, DIMY), dtype=np.float32)

    for i in range(NSAMPLES):
        # Load the dataset
        dataset = MultiModalDataset(
            sentinel1_dir='/work/mech-ai-scratch/rtali/gis-sentinel1/final_s1',
            sentinel2_dir='/work/mech-ai-scratch/rtali/gis-sentinel2/final_s2_v3',
            modis_dir='/work/mech-ai-scratch/rtali/gis-modis/modis',
            crop_dir='/work/mech-ai-scratch/rtali/gis-CDL/final_CDL',
            soil_dir='/work/mech-ai-scratch/rtali/AI_READY_IOWA/SOIL/CRSchange',
            weather_dir='/work/mech-ai-scratch/rtali/AI_READY_IOWA/WEATHER_TIFFS'
        )

        # Get channel counts
        num_channels_list = [
            len(dataset.s1_bands),      # 2 (Sentinel-1)
            len(dataset.s2_bands),      # 12 (Sentinel-2)
            len(dataset.modis_bands),   # 7 (MODIS)
            len(dataset.cdl_bands),     # Update with actual count
            len(dataset.soil_bands),     # 6 (Soil)
            len(dataset.weather_bands)   # 7 (Weather)
        ]

        print(f"Number of channels for each modality: {num_channels_list}")

        # Normalize the sample for MODIS
        X = full_normalization(dataset, 0)

        # Normalize the sample for Sentinel-1
        YS1 = full_normalization(dataset, 1)

        # Normalize the sample for Sentinel-2
        YS2 = full_normalization(dataset, 2)

        # Write the samples to the tensors
        unetX[i] = X
        unetYS1[i] = YS1
        unetYS2[i] = YS2

    print("Finished writing samples to tensors.")
    print(f"Time taken: {time.time() - start_time} seconds")

    # Check the shapes of the tensors
    print(f"unetX shape: {unetX.shape}")
    print(f"unetYS1 shape: {unetYS1.shape}")
    print(f"unetYS2 shape: {unetYS2.shape}")

    print("Saving tensors to disk...")
    start_time = time.time()

    # Save the tensors to disk as .npy files
    np.save('./models/unetX.npy', unetX)
    np.save('./models/unetYS1.npy', unetYS1)
    np.save('./models/unetYS2.npy', unetYS2)

    print("Saved tensors to disk.")
    print(f"Time taken to save tensors: {time.time() - start_time} seconds")
    print("\n======================================")
    print("All done!")
    print("Exiting...")
    print("======================================")
