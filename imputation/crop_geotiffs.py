"""
Use Anirudh's data loader for Image Imputation Dataset generation.
"""
from main_dataloader import MultiModalDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms


def normalize_channel(channel, percentile=2):
    vmin = np.percentile(channel, percentile)
    vmax = np.percentile(channel, 100 - percentile)
    channel = (channel - vmin) / (vmax - vmin)
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

    # Collect 5000 samples
    sampled_data = []
    for i, sample in enumerate(loader):
        sampled_data.append(sample)
        if i == 4999:
            break

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
    np.save('u_net_data.npy', u_net)
    print("Data saved to u_net_data.npy")
