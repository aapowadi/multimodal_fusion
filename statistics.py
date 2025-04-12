import os
import glob
import rasterio
import numpy as np
import csv
import argparse
import matplotlib.pyplot as plt

# Set up argument parser
parser = argparse.ArgumentParser(description='Calculate statistics for Sentinel-2 bands across multiple folders.')
parser.add_argument('-p', type=str, help='Path to the root folder containing subfolders with Sentinel-2 band files')
parser.add_argument('-m', type=str, default='s2', help='Name of the output CSV file (default: s2)')


# Parse the arguments
args = parser.parse_args()
output = f'{args.m}_statistics.csv'
# Validate the root folder
if not os.path.isdir(args.p):
    print(f"Error: {args.p} is not a valid directory")
    exit()

# Identify the list of bands from the first subfolder
if args.m =='soil':
    band_files = glob.glob(os.path.join(args.p, '*.tif')) 
    bands = [os.path.basename(f).split('_')[2].replace('.tif', '') for f in band_files]
    print("Bands found:", bands)
    # Initialize a list to store statistics for each band
    statistics = []
    for band in bands:
        print(f"\nProcessing band {band}") 
        file = os.path.join(args.p, f'merged_max_{band}_resampled_new.tif') 

        # Initialize variables for statistics
        min_val = float('inf')         # Current minimum
        max_val = float('-inf')        # Current maximum
        sum_val = 0.0                  # Sum of pixel values
        sum_sq_val = 0.0               # Sum of squared pixel values
        count = 0                      # Total number of pixels
        
        # Process file for the current band
        
        with rasterio.open(file) as src:
            data = src.read(1)     # Read the first (and assumed only) band
            pixels = data.flatten()  # Convert 2D array to 1D
            # Filter out negative values
            valid_pixels = pixels[pixels >= 0]
            if valid_pixels.size > 0:
                # Update statistics with this image's pixel values
                min_val = min(min_val, valid_pixels.min())
                max_val = max(max_val, valid_pixels.max())
                sum_val += np.sum(valid_pixels, dtype=np.float64)
                sum_sq_val += np.sum(valid_pixels.astype(np.float64) ** 2)
                count += valid_pixels.size

                # Plot the distribution
                plt.figure(figsize=(10, 6))
                plt.hist(valid_pixels, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
                plt.title(f'Distribution of Pixel Values for Band {band}', fontsize=14)
                plt.xlabel('Pixel Value', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                plot_file = f'distribution_{band}.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free memory
                print(f"Saved distribution plot to {plot_file}")
        
        # Compute the final statistics
        if count > 0:
            mean_val = sum_val / count
            variance = (sum_sq_val / count) - (mean_val ** 2)
            std_val = np.sqrt(variance)
            # Append the statistics for this band
            statistics.append({
                'Band': band,
                'Min': min_val,
                'Max': max_val,
                'Mean': mean_val,
                'StdDev': std_val
            })
        else:
            print(f"No pixels found for band {band}")

    # Save the statistics to a CSV file
    if statistics:
        with open(output, 'w', newline='') as csvfile:
            fieldnames = ['Band', 'Min', 'Max', 'Mean', 'StdDev']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for stat in statistics:
                writer.writerow(stat)
        print(f"Statistics saved to {output}")
    else:
        print("No statistics to save")

else:
    subfolders = [d for d in os.listdir(args.p) if os.path.isdir(os.path.join(args.p, d))]
    if not subfolders:
        print("No subfolders found")
        exit()

    first_subfolder = subfolders[0]
    band_files = glob.glob(os.path.join(args.p, first_subfolder, '4326_*.tif'))
    bands = [os.path.basename(f).split('_')[1].replace('.tif', '') for f in band_files]
    print("Bands found:", bands)

    # Initialize a list to store statistics for each band
    statistics = []

    # Process each band to compute overall statistics
    for band in bands:
        print(f"\nProcessing band {band}")
        # Find all files for this band across all subfolders  
        file_pattern = os.path.join(args.p, '*', f'4326_{band}.tif')
        files = glob.glob(file_pattern)
        if not files:
            print(f"No files found for band {band}")
            continue
        
        # Initialize variables for statistics
        min_val = float('inf')         # Current minimum
        max_val = float('-inf')        # Current maximum
        sum_val = 0.0                  # Sum of pixel values
        sum_sq_val = 0.0               # Sum of squared pixel values
        count = 0                      # Total number of pixels
        
        # Process each file for the current band
        for file in files:
            with rasterio.open(file) as src:
                data = src.read(1)     # Read the first (and assumed only) band
                pixels = data.flatten()  # Convert 2D array to 1D
                if pixels.size > 0:
                    # Update statistics with this image's pixel values
                    min_val = min(min_val, pixels.min())
                    max_val = max(max_val, pixels.max())
                    sum_val += np.sum(pixels, dtype=np.float64)
                    sum_sq_val += np.sum(pixels.astype(np.float64) ** 2)
                    count += pixels.size
        
        # Compute the final statistics
        if count > 0:
            mean_val = sum_val / count
            variance = (sum_sq_val / count) - (mean_val ** 2)
            std_val = np.sqrt(variance)
            # Append the statistics for this band
            statistics.append({
                'Band': band,
                'Min': min_val,
                'Max': max_val,
                'Mean': mean_val,
                'StdDev': std_val
            })
        else:
            print(f"No pixels found for band {band}")

    # Save the statistics to a CSV file
    if statistics:
        with open(output, 'w', newline='') as csvfile:
            fieldnames = ['Band', 'Min', 'Max', 'Mean', 'StdDev']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for stat in statistics:
                writer.writerow(stat)
        print(f"Statistics saved to {output}")
    else:
        print("No statistics to save")