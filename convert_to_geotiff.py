import os
import xarray as xr
import rioxarray
import rasterio
from datetime import datetime, timedelta
from collections import defaultdict

def preprocess_weather_to_weekly_geotiffs(sentinel1_dir, weather_dir, output_dir):
    """
    Convert weather NetCDF files to weekly mean GeoTIFFs for April to September,
    with each GeoTIFF named by week_start_date and stacking variables as channels.

    Args:
        sentinel1_dir (str): Directory containing Sentinel-1 folders (YYYY-MM-DD).
        weather_dir (str): Directory containing IA_year.nc files.
        output_dir (str): Directory to save the weekly mean GeoTIFFs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all week_start_dates from Sentinel-1, filtered for April to September
    week_start_dates = [
        d for d in os.listdir(sentinel1_dir)
        if os.path.isdir(os.path.join(sentinel1_dir, d))
    ]
    dates_by_year = defaultdict(list)
    for date_str in week_start_dates:
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month = date.month
            day = date.day
            if (month == 4 and day >= 1) or (4 < month < 9) or (month == 9 and day <= 30):
                year = date.year
                dates_by_year[year].append(date_str)
        except ValueError:
            continue

    # Process each year
    for year, dates in dates_by_year.items():
        nc_path = os.path.join(weather_dir, f'IA_{year}.nc')
        if not os.path.exists(nc_path):
            print(f"Warning: No weather file for year {year}")
            continue
        ds = xr.open_dataset(nc_path)
        # Ensure CRS is set
        if not ds.rio.crs:
            ds = ds.rio.write_crs("epsg:4326")

        # Process each week
        for date_str in dates:
            start_date = datetime.strptime(date_str, '%Y-%m-%d')
            end_date = start_date + timedelta(days=6)
            # Select data for the week
            ds_week = ds.sel(time=slice(start_date, end_date))
            if len(ds_week['time']) == 0:
                print(f"No data for week starting {date_str}")
                continue
            # Compute mean over time for all variables
            mean_ds = ds_week.mean(dim='time')
            variables = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
            mean_da = mean_ds[variables].to_array(dim='variable')
            # Write to GeoTIFF
            output_path = os.path.join(output_dir, f'{date_str}.tif')
            data = mean_da.values  # Shape: (7, height, width)
            crs = mean_da.rio.crs
            transform = mean_da.rio.transform()
            height, width = mean_da.shape[1], mean_da.shape[2]
            variables = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']

            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': len(variables),
                'dtype': data.dtype,
                'crs': crs,
                'transform': transform,
                'descriptions': variables
            }

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
            print(f"Saved weekly mean GeoTIFF for {date_str}")

# Example usage
sentinel1_dir = '/work/mech-ai-scratch/rtali/gis-sentinel1/final_s1'
weather_dir = '/work/mech-ai-scratch/rtali/AI_READY_IOWA/WEATHER'
output_dir = '/work/mech-ai-scratch/rtali/AI_READY_IOWA/WEATHER_TIFFS'
preprocess_weather_to_weekly_geotiffs(sentinel1_dir, weather_dir, output_dir)