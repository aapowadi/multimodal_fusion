import os
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import rasterio
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datetime import datetime
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    def __init__(self, sentinel1_dir, sentinel2_dir, modis_dir, crop_dir, soil_dir, weather_dir, transform=None):
        self.sentinel1_dir = sentinel1_dir
        self.sentinel2_dir = sentinel2_dir
        self.modis_dir = modis_dir
        self.cdl_dir = crop_dir
        self.soil_dir = soil_dir
        self.weather_dir = weather_dir
        self.transform = transform
        self.patch_size = 16
        self.delta_lat = 0.01446

        self.week_start_dates = [
            d for d in os.listdir(sentinel1_dir)
            if os.path.isdir(os.path.join(sentinel1_dir, d)) and self._is_in_april_to_september(d) 
        ]
        if not self.week_start_dates:
            raise ValueError("No Sentinel-1 data found for April to September.")

        self.s1_bands = ['vv', 'vh']
        self.s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] # 12 bands + 'B10' which is missing sometimes.
        self.modis_bands = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7']
        self.weather_bands = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
        self.cdl_bands = ['Band_1']
        self.soil_bands = ['aws100', 'aws150', 'aws999', 'nccpi3all', 'nccpi3corn', 'rootznaws', 'soc150', 'soc999', 'pctearthmc']

        sample_s1_path = os.path.join(sentinel1_dir, self.week_start_dates[0], '4326_vv.tif')
        with rasterio.open(sample_s1_path) as src:
            self.bounds = src.bounds
            self.transform_geo = src.transform
            self.width, self.height = src.width, src.height

    def _is_in_april_to_september(self, date_str):
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month = date.month
            day = date.day
            return (month == 4 and day >= 1) or (4 < month < 9) or (month == 9 and day <= 23)
        except ValueError:
            return False

    def _load_and_crop_single_band(self, path, bbox, cdl=False, band_name=None, stats=None):
        try:
            with rasterio.open(path) as src:
                window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
                data = src.read(1, window=window, boundless=True, fill_value=np.nan)
                data_tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
                interpolation = F.InterpolationMode.NEAREST_EXACT if cdl else F.InterpolationMode.BILINEAR
                resized = F.resize(data_tensor, [self.patch_size, self.patch_size], interpolation=interpolation, antialias=True)
                return resized.squeeze(0)
        except Exception as e:
            logger.warning(f"Failed to load band {band_name} from {path}: {e}. Generating dummy data.")
            # Generate dummy data
            if stats and band_name in stats:
                vmin, vmax = stats[band_name]
                dummy_value = (vmin + vmax) / 2
            else:
                dummy_value = 0.0
            return torch.full((self.patch_size, self.patch_size), dummy_value, dtype=torch.float32)

    def _random_patch_coords(self):
        sample_path = os.path.join(self.sentinel1_dir, self.week_start_dates[0], '4326_vv.tif')
        with rasterio.open(sample_path) as src:
            bounds = src.bounds
        
        lon_min = bounds.left + self.delta_lat / 2
        lon_max = bounds.right - self.delta_lat / 2
        lat_min = bounds.bottom + self.delta_lat / 2
        lat_max = bounds.top - self.delta_lat / 2
        
        lat_center = np.random.uniform(lat_min, lat_max)
        lon_center = np.random.uniform(lon_min, lon_max)
        
        delta_lat = self.delta_lat
        delta_lon = delta_lat / np.cos(np.radians(lat_center))
        
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
        bbox = self._random_patch_coords()

        s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max = load_statistics()

        # Sentinel-1
        s1_folder = os.path.join(self.sentinel1_dir, week_start_date)
        s1_bands = []
        # print('Loading S1')
        for band in self.s1_bands:
            band_ar = self._load_and_crop_single_band(os.path.join(s1_folder, f'4326_{band}.tif'), bbox, band_name=band, stats=s1_min_max)
            vmin, vmax = s1_min_max[band]
            channel = (band_ar - vmin) / (vmax - vmin + 1e-8)
            s1_bands.append(channel)

        s1_tensor = torch.stack(s1_bands, dim=0)
        # print(f'shape:{s1_tensor.shape}')
        # Sentinel-2
        s2_folder = os.path.join(self.sentinel2_dir, week_start_date)
        s2_bands = []
        # print('Loading S2')
        for band in self.s2_bands:
            band_ar = self._load_and_crop_single_band(os.path.join(s2_folder, f'4326_{band}.tif'), bbox, band_name=band, stats=s2_min_max)
            vmin, vmax = s2_min_max[band]
            channel = (band_ar - vmin) / (vmax - vmin + 1e-8)
            s2_bands.append(channel)

        s2_tensor = torch.stack(s2_bands, dim=0)
        # print(f'shape:{s2_tensor.shape}')
        # MODIS
        modis_folder = os.path.join(self.modis_dir, week_start_date)
        modis_bands = []
        # print('Loading Modis')
        for band in self.modis_bands:
            band_ar = self._load_and_crop_single_band(
                os.path.join(modis_folder, f'4326_{band}.tif'), bbox, band_name=band, stats=modis_min_max)
            vmin, vmax = modis_min_max[band]
            channel = (band_ar - vmin) / (vmax - vmin + 1e-8)
            modis_bands.append(channel)
        modis_tensor = torch.stack(modis_bands, dim=0)
        # print(f'shape:{modis_tensor.shape}')
        # CDL (used for label)
        crop_path = os.path.join(self.cdl_dir, f'{year}_WGS84.tif')
        # print('Loading CDL')
        cdl_tensor = self._load_and_crop_single_band(crop_path, bbox, cdl=True, band_name='Band_1')
        cdl_tensor = torch.reshape(cdl_tensor,(-1,cdl_tensor.shape[0],cdl_tensor.shape[1]))
        # print(f'shape:{cdl_tensor.shape}')
        # Soil
        soil_bands = []
        # print('Loading Soil')
        for band in self.soil_bands:
            band_ar = self._load_and_crop_single_band(
                os.path.join(self.soil_dir, f'merged_max_{band}_resampled_new.tif'), bbox, band_name=band, stats=soil_min_max)
            vmin, vmax = soil_min_max[band]
            channel = (band_ar - vmin) / (vmax - vmin + 1e-8)
            soil_bands.append(channel)

        soil_tensor = torch.stack(soil_bands, dim=0)
        # print(f'shape:{soil_tensor.shape}')
        # Weather
        weather_patches = []
        # print('Loading Weather')
        date = datetime.strptime(week_start_date, '%Y-%m-%d')
        doy = date.timetuple().tm_yday
        for band in self.weather_bands:
            week_stack = []
            for i in range(doy, doy + 7):
                date_s = datetime.strptime(f'{year}-{i}', '%Y-%j')
                date_str = date_s.strftime('%Y-%m-%d')
                path = os.path.join(self.weather_dir, band, f'{date_str}.tif')
                try:
                    band_ar = self._load_and_crop_single_band(path, bbox, band_name=band, stats=weather_min_max)
                    vmin, vmax = weather_min_max[band]
                    channel = (band_ar - vmin) / (vmax - vmin + 1e-8)
                    week_stack.append(channel)
                except Exception as e:
                    logger.warning(f"Skipping missing weather band {band} for {date_str}: {e}")
                    continue
            if week_stack:
                week_avg = torch.nanmean(torch.stack(week_stack, dim=0), dim=0)
            else:
                logger.warning(f"All daily files missing for weather band {band}. Generating dummy data.")
                dummy_value = weather_min_max.get(band, (0.0, 0.0))[0] if weather_min_max else 0.0
                week_avg = torch.full((self.patch_size, self.patch_size), dummy_value, dtype=torch.float32)
            weather_patches.append(week_avg)
        weather_tensor = torch.stack(weather_patches, dim=0)
        # print(f'shape:{weather_tensor.shape}')
        s1_tensor = torch.cat((s1_tensor,weather_tensor,soil_tensor,cdl_tensor), dim=0)
        s2_tensor = torch.cat((s2_tensor,weather_tensor,soil_tensor,cdl_tensor), dim=0)
        modis_tensor = torch.cat((modis_tensor,weather_tensor,soil_tensor,cdl_tensor), dim=0)

        return [s1_tensor, s2_tensor, modis_tensor]


def load_statistics(s1_path='s1_statistics.csv', s2_path='s2_statistics.csv', modis_path='modis_statistics.csv',
                    soil_path='soil_statistics.csv', weather_path='weather_statistics.csv'):
    try:
        s1_stats = pd.read_csv(os.path.join('statistics', s1_path))
        s2_stats = pd.read_csv(os.path.join('statistics', s2_path))
        modis_stats = pd.read_csv(os.path.join('statistics', modis_path))
        soil_stats = pd.read_csv(os.path.join('statistics', soil_path))
        weather_stats = pd.read_csv(os.path.join('statistics', weather_path))
    except FileNotFoundError as e:
        logger.error(f"Could not find one of the CSV files: {e}")
        return None, None, None, None, None
    
    s1_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in s1_stats.iterrows()}
    s2_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in s2_stats.iterrows()}
    modis_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in modis_stats.iterrows()}
    soil_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in soil_stats.iterrows()}
    weather_min_max = {row['Band']: (row['Min'], row['Max']) for _, row in weather_stats.iterrows()}
    
    return s1_min_max, s2_min_max, modis_min_max, soil_min_max, weather_min_max