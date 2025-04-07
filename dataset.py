import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import rasterio
import torchvision.transforms as transforms

class MultiModalDataset(Dataset):
    def __init__(self, samples, sentinel1_dir, sentinel2_dir, modis_dir, crop_dir, soil_file, weather_dir, transform=None):
        """
        Initialize the MultiModalDataset with paths to data directories and a list of samples.

        Args:
            samples (list): List of tuples (week_start_date, x, y, label), where week_start_date is a string (e.g., '2023_01'),
                            x and y are pixel coordinates, and label is the target.
            sentinel1_dir (str): Directory containing Sentinel-1 data in week_start_date/vv.tif, vh.tif format.
            sentinel2_dir (str): Directory containing Sentinel-2 data in week_start_date/B01.tif, B02.tif, etc. format.
            modis_dir (str): Directory containing MODIS data in week_start_date/Band1.tif, Band2.tif, etc. format.
            crop_dir (str): Directory containing crop data netCDF files (IA_year.nc).
            soil_file (str): Path to the soil data netCDF file (IA.nc).
            weather_dir (str): Directory containing weather data netCDF files (IA_year.nc).
            transform (callable, optional): Optional transform to be applied to the data.
        """
        self.samples = samples  # List of (week_start_date, x, y, label)
        self.sentinel1_dir = sentinel1_dir
        self.sentinel2_dir = sentinel2_dir
        self.modis_dir = modis_dir
        self.crop_dir = crop_dir
        self.soil_file = soil_file
        self.weather_dir = weather_dir
        self.transform = transform
        self.patch_size = 224  # Size of the patch to extract (224x224)

        # Load static soil data into memory
        self.soil_ds = xr.open_dataset(soil_file)

        # Load crop and weather datasets per year (lazily using xarray)
        self.crop_ds = {}
        self.weather_ds = {}
        unique_years = set(date.split('_')[0] for date, _, _, _ in samples)  # Extract year from week_start_date
        for year in unique_years:
            crop_file = os.path.join(crop_dir, f'IA_{year}.nc')
            weather_file = os.path.join(weather_dir, f'IA_{year}.nc')
            self.crop_ds[year] = xr.open_dataset(crop_file)
            self.weather_ds[year] = xr.open_dataset(weather_file)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (x, label), where x is a tensor of shape (6, 3, 224, 224) containing all modalities,
                   and label is the target.
        """
        week_start_date, x, y, label = self.samples[idx]
        year = week_start_date.split('_')[0]  # Extract year from week_start_date (e.g., '2023' from '2023_01')

        # Define patch boundaries (centered at (x, y))
        half_patch = self.patch_size // 2
        y_start, y_end = y - half_patch, y + half_patch
        x_start, x_end = x - half_patch, x + half_patch

        ### 1. Sentinel-1: Load vv.tif and vh.tif, extract patch, map to 3 channels
        s1_folder = os.path.join(self.sentinel1_dir, week_start_date)
        vv_path = os.path.join(s1_folder, 'vv.tif')
        vh_path = os.path.join(s1_folder, 'vh.tif')
        with rasterio.open(vv_path) as src:
            vv_patch = src.read(1)[y_start:y_end, x_start:x_end]  # Shape: (224, 224)
        with rasterio.open(vh_path) as src:
            vh_patch = src.read(1)[y_start:y_end, x_start:x_end]
        s1_tensor = torch.stack([
            torch.from_numpy(vv_patch).float(),
            torch.from_numpy(vh_patch).float()
        ], dim=0)  # Shape: (2, 224, 224)
        # Map 2 channels to 3 by repeating the first channel
        s1_tensor = torch.cat([s1_tensor, s1_tensor[:1]], dim=0)  # Shape: (3, 224, 224)

        ### 2. Sentinel-2: Load selected bands (e.g., B02, B03, B04 for RGB), extract patch
        s2_folder = os.path.join(self.sentinel2_dir, week_start_date)
        s2_bands = ['B02', 'B03', 'B04']  # Example: select RGB-like bands
        s2_patches = []
        for band in s2_bands:
            band_path = os.path.join(s2_folder, f'{band}.tif')
            with rasterio.open(band_path) as src:
                band_patch = src.read(1)[y_start:y_end, x_start:x_end]
            s2_patches.append(torch.from_numpy(band_patch).float())
        s2_tensor = torch.stack(s2_patches, dim=0)  # Shape: (3, 224, 224)

        ### 3. MODIS: Load selected bands (e.g., Band1, Band2, Band3), extract patch
        modis_folder = os.path.join(self.modis_dir, week_start_date)
        modis_bands = ['Band1', 'Band2', 'Band3']  # Select first 3 bands
        modis_patches = []
        for band in modis_bands:
            band_path = os.path.join(modis_folder, f'{band}.tif')
            with rasterio.open(band_path) as src:
                band_patch = src.read(1)[y_start:y_end, x_start:x_end]
            modis_patches.append(torch.from_numpy(band_patch).float())
        modis_tensor = torch.stack(modis_patches, dim=0)  # Shape: (3, 224, 224)

        ### 4. Crop Data: Extract value for the week and location, repeat to 3 channels
        crop_ds = self.crop_ds[year]
        # Assume 'time' is a datetime coordinate and matches week_start_date
        time_idx = np.where(crop_ds['time'].values == np.datetime64(week_start_date))[0][0]
        crop_value = float(crop_ds['crop_variable'].isel(time=time_idx, x=x, y=y).values)  # Replace 'crop_variable' with actual name
        crop_tensor = torch.full((1, 224, 224), crop_value, dtype=torch.float)
        crop_tensor = crop_tensor.repeat(3, 1, 1)  # Shape: (3, 224, 224)

        ### 5. Soil Data: Extract static value at location, repeat to 3 channels
        soil_value = float(self.soil_ds['soil_variable'].isel(x=x, y=y).values)  # Replace 'soil_variable' with actual name
        soil_tensor = torch.full((1, 224, 224), soil_value, dtype=torch.float)
        soil_tensor = soil_tensor.repeat(3, 1, 1)  # Shape: (3, 224, 224)

        ### 6. Weather Data: Extract values for the week and location, map to 3 channels
        weather_ds = self.weather_ds[year]
        time_idx = np.where(weather_ds['time'].values == np.datetime64(week_start_date))[0][0]
        # Assume weather variables are 'temp', 'precip', 'humidity' (replace with actual names)
        weather_vars = ['temp', 'precip', 'humidity']
        weather_values = [float(weather_ds[var].isel(time=time_idx, x=x, y=y).values) for var in weather_vars]
        weather_tensor = torch.stack([
            torch.full((224, 224), val, dtype=torch.float) for val in weather_values
        ], dim=0)  # Shape: (3, 224, 224)

        # Stack all modalities into a single tensor
        modalities = [s1_tensor, s2_tensor, modis_tensor, crop_tensor, soil_tensor, weather_tensor]
        x = torch.stack(modalities, dim=0)  # Shape: (6, 3, 224, 224)

        # Apply any additional transforms (e.g., normalization)
        if self.transform:
            x = self.transform(x)

        return x, label
    
    # Example samples
samples = [('2023_01', 100, 150, 0), ('2023_02', 200, 250, 1)]

# Paths to data
dataset = MultiModalDataset(
    samples=samples,
    sentinel1_dir='/work/mech-ai-scratch/rtali/gis-sentinel1/final_s1',
    sentinel2_dir='/work/mech-ai-scratch/rtali/gis-sentinel2/final_s2_v3',
    modis_dir='/work/mech-ai-scratch/rtali/gis-modis/final_modis_data',
    crop_dir='/work/mech-ai-scratch/rtali/AI_READY_IOWA/CDL/IN4326',
    soil_file='/work/mech-ai-scratch/rtali/AI_READY_IOWA/SOIL',
    weather_dir='/work/mech-ai-scratch/rtali/AI_READY_IOWA/WEATHER',
    transform=transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Example transform
)

# Access a sample
x, label = dataset[0]
print(x.shape)  # torch.Size([6, 3, 224, 224])
print(label)    # e.g., 0