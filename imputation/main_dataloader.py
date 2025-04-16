import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import rasterio
import torchvision.transforms as transforms
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from glob import glob
import pandas as pd


class MultiModalDataset(Dataset):
    def __init__(self, sentinel1_dir, sentinel2_dir, modis_dir, crop_dir, soil_dir, weather_dir, transform=None):
        """
        Initialize the dataset using Sentinel-1 folder dates (YYYY-MM-DD) and random 16x16 patches
        from a 1 mile x 1 mile area in WGS84 degrees.

        Args:
            sentinel1_dir (str): Directory for Sentinel-1 data (YYYY-MM-DD/vv.tif, vh.tif).
            sentinel2_dir (str): Directory for Sentinel-2 data (YYYY-MM-DD/B01.tif, etc.).
            modis_dir (str): Directory for MODIS data (YYYY-MM-DD/Band1.tif, etc.).
            crop_dir (str): Directory containing crop data NetCDF files (IA_year.nc).
            soil_file (str): Path to the soil NetCDF file (e.g., IA.nc) with variables 'nccpi3all', etc.
            weather_dir (str): Directory containing weather data NetCDF files (IA_year.nc).
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.sentinel1_dir = sentinel1_dir
        self.sentinel2_dir = sentinel2_dir
        self.modis_dir = modis_dir
        self.cdl_dir = crop_dir
        self.soil_dir = soil_dir
        self.weather_dir = weather_dir
        self.transform = transform
        self.patch_size = 16*4  # 16x16 patch size
        self.delta_lat = 0.01446*4  # 1 mile in latitude degrees (approx.)

        # Get week_start_dates from Sentinel-1 folder, filter for April to September
        self.week_start_dates = [
            d for d in os.listdir(sentinel1_dir)
            if os.path.isdir(os.path.join(sentinel1_dir, d)) and self._is_in_april_to_september(d)
        ]
        if not self.week_start_dates:
            raise ValueError(
                "No Sentinel-1 data found for April to September.")

        # Define all bands/variables
        self.s1_bands = ['vv', 'vh']  # 2 bands
        # 12 bands + 'B10' which is missing sometimes.
        self.s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05',
                         'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        self.modis_bands = ['Band1', 'Band2', 'Band3',
                            'Band4', 'Band5', 'Band6', 'Band7']  # 7 bands
        self.weather_bands = ['dayl', 'prcp', 'srad',
                              'swe', 'tmax', 'tmin', 'vp']  # 7 variables
        # Crop Data Layer: 120 classes representing land cover uses including different crops like 'Maize', 'Soy', etc.
        self.cdl_bands = ['Band_1']
        self.soil_bands = ['aws100', 'aws150', 'aws999', 'nccpi3all', 'nccpi3corn',
                           'rootznaws', 'soc150', 'soc999', 'pctearthmc']  # 6 variables
        # Get spatial bounds from a sample Sentinel-1 GeoTIFF
        sample_s1_path = os.path.join(
            sentinel1_dir, self.week_start_dates[0], '4326_vv.tif')
        with rasterio.open(sample_s1_path) as src:
            self.bounds = src.bounds  # (left, bottom, right, top) in degrees
            self.transform_geo = src.transform  # Geotransform to convert degrees to pixels
            self.width, self.height = src.width, src.height

    
    def _is_in_april_to_september(self, date_str):
        """Check if date_str (YYYY-MM-DD) is between April 1st and September 30th."""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month = date.month
            day = date.day
            return (month == 4 and day >= 1) or (4 < month < 9) or (month == 9 and day <= 30)
        except ValueError:
            return False

    def _load_and_crop_single_band(self, path, bbox, cdl=True):
        """Load and crop a single-band GeoTIFF using coordinate bounds, resize to 16x16."""
        with rasterio.open(path) as src:
            window = rasterio.windows.from_bounds(
                *bbox, transform=src.transform)
            data = src.read(1, window=window, boundless=True,
                            fill_value=np.nan)  # Handle out-of-bounds
            data_tensor = torch.from_numpy(
                data).float().unsqueeze(0)  # (1, height, width)
            if cdl == True:
                resized = F.resize(data_tensor, [
                                   self.patch_size, self.patch_size], interpolation=F.InterpolationMode.NEAREST_EXACT)
            else:
                resized = F.resize(data_tensor, [
                                   self.patch_size, self.patch_size], interpolation=F.InterpolationMode.BILINEAR)
            return resized.squeeze(0)  # (16, 16)

    def _load_and_crop_multi_band(self, path, bbox):
        """Load and crop a multi-band GeoTIFF using coordinate bounds, resize to 16x16."""
        with rasterio.open(path) as src:
            window = rasterio.windows.from_bounds(
                *bbox, transform=src.transform)
            data = src.read(window=window, boundless=True,
                            fill_value=np.nan)  # (bands, height, width)
            data_tensor = torch.from_numpy(
                data).float()  # (bands, height, width)
            resized = F.resize(data_tensor, [
                               self.patch_size, self.patch_size], interpolation=F.InterpolationMode.BILINEAR)
            return resized  # (bands, 16, 16)

    def _random_patch_coords(self):
        """Generate random center and calculate 1-mile x 1-mile bounding box in degrees."""
        # Get bounds from a sample GeoTIFF, e.g., Sentinel-1
        sample_path = os.path.join(
            self.sentinel1_dir, self.week_start_dates[0], '4326_vv.tif')
        with rasterio.open(sample_path) as src:
            bounds = src.bounds  # (left, bottom, right, top) in degrees

        # Random center within bounds, leaving room for 1-mile extent
        lon_min = bounds.left + self.delta_lat / 2
        lon_max = bounds.right - self.delta_lat / 2
        lat_min = bounds.bottom + self.delta_lat / 2
        lat_max = bounds.top - self.delta_lat / 2

        lat_center = np.random.uniform(lat_min, lat_max)
        lon_center = np.random.uniform(lon_min, lon_max)

        # Calculate 1-mile extents
        delta_lat = self.delta_lat  # Fixed for latitude
        # Adjust for latitude
        delta_lon = delta_lat / np.cos(np.radians(lat_center))

        # Define bounding box
        left = lon_center - delta_lon / 2
        right = lon_center + delta_lon / 2
        bottom = lat_center - delta_lat / 2
        top = lat_center + delta_lat / 2

        return left, bottom, right, top

    def __len__(self):
        return len(self.week_start_dates)

    def __getitem__(self, idx):
        week_start_date = self.week_start_dates[idx]
        year = week_start_date.split('-')[0]
        week_start_date = self.week_start_dates[idx]
        bbox = self._random_patch_coords()  # (left, bottom, right, top)

        # Sentinel-1 (separate files)
        s1_folder = os.path.join(self.sentinel1_dir, week_start_date)
        s1_patches = []
        for band in self.s1_bands:
            path = os.path.join(s1_folder, f'4326_{band}.tif')
            patch = self._load_and_crop_single_band(path, bbox)
            s1_patches.append(patch)
        s1_tensor = torch.stack(s1_patches, dim=0)  # (2, 16, 16)

        # Sentinel-2 (separate files)
        s2_folder = os.path.join(self.sentinel2_dir, week_start_date)
        s2_patches = []
        for band in self.s2_bands:
            path = os.path.join(s2_folder, f'4326_{band}.tif')
            patch = self._load_and_crop_single_band(path, bbox)
            s2_patches.append(patch)
        s2_tensor = torch.stack(s2_patches, dim=0)  # (13, 16, 16)

        # MODIS (separate files)
        modis_folder = os.path.join(self.modis_dir, week_start_date)
        modis_patches = []
        for band in self.modis_bands:
            path = os.path.join(modis_folder, f'4326_{band}.tif')
            patch = self._load_and_crop_single_band(path, bbox)
            modis_patches.append(patch)
        modis_tensor = torch.stack(modis_patches, dim=0)  # (7, 16, 16)

        # Crop (single-band)
        crop_path = os.path.join(self.cdl_dir, f'{year}_WGS84.tif')
        cdl_tensor = self._load_and_crop_single_band(
            crop_path, bbox).unsqueeze(0)  # (1, 16, 16)

        # 4. soil data (all bands)
        soil_patches = []
        for band in self.soil_bands:
            path = os.path.join(
                self.soil_dir, f'merged_max_{band}_resampled_new.tif')
            patch = self._load_and_crop_single_band(path, bbox)
            soil_patches.append(patch)
        soil_tensor = torch.stack(soil_patches, dim=0)  # (6, 16, 16)

        # Weather data (separate files)
        # cropped_ds = ds.sel(
        # lon=slice(bbox['left'], bbox['right']),
        # lat=slice(bbox['bottom'], bbox['top']))
        weather_patches = []
        date = datetime.strptime(f'{week_start_date}', '%Y-%m-%d')
        doy = date.timetuple().tm_yday
        for band in self.weather_bands:
            week_stack = []
            for i in range(doy, doy+7):
                date_s = datetime.strptime(f'{year}-{i}', '%Y-%j')
                year = date_s.strftime("%Y")  # Gets "2023"
                month = date_s.strftime("%m")  # Gets "01"
                day = date_s.strftime("%d")   # Gets "05"
                date_str = f"{year}-{month}-{day}"
                path = os.path.join(self.weather_dir, band, f'{date_str}.tif')
                patch = self._load_and_crop_single_band(path, bbox)
                week_stack.append(patch)
            # Average over 7 days of all non-NaN values
            week_avg = torch.nanmean(torch.stack(week_stack, dim=0), dim=0)
            weather_patches.append(week_avg)
        weather_tensor = torch.stack(weather_patches, dim=0)  # (7, 16, 16)

        # Stack modalities into a list (variable channels per modality)
        modalities = [s1_tensor, s2_tensor, modis_tensor,
                      cdl_tensor, soil_tensor, weather_tensor]
        # Shapes: [(2, 16, 16), (13, 16, 16), (7, 16, 16), (N_crop, 16, 16), (6, 16, 16), (7, 16, 16)]

        if self.transform:
            modalities = [self.transform(m) for m in modalities]

        # Dummy label (replace with actual labels if available)
        label = 0

        return modalities, label
        
        
