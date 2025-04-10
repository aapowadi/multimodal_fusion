import xarray as xr
import pandas as pd
import os
import glob
import numpy as np

# Specify the folder containing .nc files
folder_path = '/work/mech-ai-scratch/rtali/AI_READY_IOWA/WEATHER'

# Get list of all .nc files in the folder
nc_files = glob.glob(os.path.join(folder_path, '*.nc'))

# Initialize lists to store statistics
stats_data = {
    'Band': [],
    'Min': [],
    'Max': [],
    'Mean': [],
    'StdDev': []
}

# Check if files exist
if not nc_files:
    print("No .nc files found in the specified folder.")
else:
    # Open all .nc files as a multi-file dataset
    ds = xr.open_mfdataset(nc_files, combine='by_coords')
    
    # Filter for April to September (months 4 to 9)
    ds = ds.sel(time=ds.time.dt.month.isin([4, 5, 6, 7, 8, 9]))
    ds = ds.drop_vars(['lat','lon'])
    # Get variable names (excluding coordinate variables)
    data_vars = [var for var in ds.data_vars if var not in ds.coords]
    
    if len(data_vars) < 7:
        print(f"Warning: Found only {len(data_vars)} variables instead of 7")    
    # Calculate statistics for each variable
    for var in data_vars:
        print(f"Processing variable: {var}")
        
        # Extract the variable data
        data = ds[var]
        
        # Calculate statistics
        # Convert to numpy array and flatten for calculations
        data_array = data.values
        if data_array.size > 0:  # Check if data is not empty
            min_val = np.nanmin(data_array)
            max_val = np.nanmax(data_array)
            mean_val = np.nanmean(data_array)
            std_val = np.nanstd(data_array)
            
            # Append to stats_data
            stats_data['Band'].append(var)
            stats_data['Min'].append(min_val)
            stats_data['Max'].append(max_val)
            stats_data['Mean'].append(mean_val)
            stats_data['StdDev'].append(std_val)
        else:
            print(f"Warning: No valid data for variable {var} in April-September")
            # Append NaN values for empty data
            stats_data['Band'].append(var)
            stats_data['Min'].append(np.nan)
            stats_data['Max'].append(np.nan)
            stats_data['Mean'].append(np.nan)
            stats_data['StdDev'].append(np.nan)
    
    # Close the dataset
    ds.close()

# Create DataFrame from statistics
stats_df = pd.DataFrame(stats_data)

# Save to CSV
output_csv = 'weather_statistics.csv'
stats_df.to_csv(output_csv, index=False)
print(f"Statistics saved to {output_csv}")

# Print the results
print("\nCalculated Statistics (April-September):")
print(stats_df)