import os
import glob
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Calculate statistics for Sentinel-2 bands across multiple folders.')
parser.add_argument('-p', type=str, help='Path to the root folder containing subfolders with Sentinel-2 band files')
parser.add_argument('-m', type=str, default='s2', help='Name of the output CSV file (default: s2)')


# Parse the arguments
args = parser.parse_args()
csv_file = f'{args.m}_statistics.csv'
# Validate inputs
if not os.path.isdir(args.p):
    print(f"Error: {args.p} is not a valid directory")
    exit()
if not os.path.isfile(csv_file):
    print(f"Error: {csv_file} is not a valid file")
    exit()

# Create output directory if it doesn't exist
output_dir = 'data_plots'
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)
bands = df['Band'].tolist()
print("Bands found in CSV:", bands)

# Process each band to gather pixel data and plot distributions
for band in bands:
    print(f"\nProcessing distribution for band {band}")
    # Find all files for this band across all subfolders
    file_pattern = os.path.join(args.p, '*', f'4326_{band}.tif')
    files = glob.glob(file_pattern)
    if not files:
        print(f"No files found for band {band}")
        continue
    
    # Collect all pixel values for this band
    all_pixels = []
    for file in files:
        with rasterio.open(file) as src:
            data = src.read(1)  # Read the first (and assumed only) band
            pixels = data.flatten()  # Convert 2D array to 1D
            if pixels.size > 0:
                all_pixels.extend(pixels)
    
    if not all_pixels:
        print(f"No pixel data found for band {band}")
        continue
    
    # Convert to numpy array for efficient computation
    pixel_array = np.array(all_pixels)
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(pixel_array, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of Pixel Values for Band {band}', fontsize=14)
    plt.xlabel('Pixel Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics from CSV to the plot
    band_stats = df[df['Band'] == band].iloc[0]
    stats_text = (f"Min: {band_stats['Min']:.2f}\n"
                  f"Max: {band_stats['Max']:.2f}\n"
                  f"Mean: {band_stats['Mean']:.2f}\n"
                  f"StdDev: {band_stats['StdDev']:.2f}")
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the plot
    output_file = os.path.join(output_dir, f'distribution_{band}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Saved plot to {output_file}")

print("\nAll distributions plotted and saved.")