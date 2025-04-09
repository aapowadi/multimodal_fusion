import xarray as xr
import rioxarray
import os
 
def convert_weather_to_geotiff(weather_dir, output_dir):
    """
    Convert weather NetCDF files to GeoTIFFs for each variable and time step within April to September.

    Args:
        weather_dir (str): Directory containing IA_year.nc files.
        output_dir (str): Directory to save the GeoTIFFs, organized by variable.
    """
    for year_file in os.listdir(weather_dir):

        if year_file.endswith('.nc'):
            year = year_file.split('_')[1].split('.')[0]  # e.g., '2023' from 'IA_2023.nc'
            ds = xr.open_dataset(os.path.join(weather_dir, year_file))
            # Set CRS if not already set
            if not ds.rio.crs:
                ds = ds.rio.write_crs("epsg:4326")
            # Select time slice from April 1 to September 30
            start_date = f'{year}-04-01'
            end_date = f'{year}-09-30'
            ds_subset = ds.sel(time=slice(start_date, end_date))
            for time in ds_subset['time']:
                date_str = str(time.values)[:10]  # YYYY-MM-DD
                for var in ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']:
                    da = ds_subset[var].sel(time=time)
                    var_dir = os.path.join(output_dir, var)
                    os.makedirs(var_dir, exist_ok=True)
                    output_path = os.path.join(var_dir, f'{date_str}.tif')
                    da.rio.to_raster(output_path)
                    print(f"Saved {output_path}")

weather_dir = '/work/mech-ai-scratch/rtali/AI_READY_IOWA/WEATHER'
output_dir = '/work/mech-ai-scratch/rtali/AI_READY_IOWA/WEATHER_TIFFS'
convert_weather_to_geotiff(weather_dir, output_dir)