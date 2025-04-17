"""
The idea is to take all Sentinel-1 and Sentinel-2 images for every week of the year, one band at a time. 
Calculate Union of all the images for each band. Stack them together and save them as a single image.
This will be done for each year.
"""

import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import glob
from curve_fitting import impute_missing_values
from numba import njit, prange


def aggregate_bands_by_year(input_files):
    # Step 1: Read metadata and calculate union bounds
    all_bounds = []
    resolutions = []

    with rasterio.open(input_files[0]) as ref:
        dst_crs = ref.crs  # Use CRS of first file

    # Gather bounds and resolutions
    for f in input_files:
        with rasterio.open(f) as src:
            if src.crs != dst_crs:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                bounds = rasterio.transform.array_bounds(
                    height, width, transform)
            else:
                bounds = src.bounds
            all_bounds.append(bounds)
            resolutions.append((src.res[0], src.res[1]))

    # Calculate union bounds and the coarsest resolution
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
    res_x = max(r[0] for r in resolutions)
    res_y = max(r[1] for r in resolutions)

    # Target transform and shape
    dst_width = int((maxx - minx) / res_x)
    dst_height = int((maxy - miny) / res_y)
    dst_transform = from_bounds(minx, miny, maxx, maxy, dst_width, dst_height)

    # Step 2: Reproject all rasters onto the new grid
    output_array = np.full((len(input_files), dst_height,
                            dst_width), np.nan, dtype=np.float32)

    for i, f in enumerate(input_files):
        with rasterio.open(f) as src:
            src_data = src.read(1)
            temp_array = np.full((dst_height, dst_width),
                                 np.nan, dtype=np.float32)

            reproject(
                source=src_data,
                destination=temp_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
            output_array[i] = temp_array

    # Step 3: Save to output GeoTIFF
    meta = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": len(input_files),
        "dtype": 'float32',
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "lzw"
    }

    with rasterio.open("aligned_union_multiband.tif", "w", **meta) as dst:
        for i in range(len(input_files)):
            dst.write(output_array[i], i + 1)


def obtain_files(years, s1_dir, s2_dir):

    s1_bands = ["vv", "vh"]
    s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05',
                'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    all_s1_files = {}
    all_s2_files = {}

    for yr in years:

        print(f"Processing year {yr}")

        # Get all Sentinel-1 and Sentinel-2 files for the year.

        all_s1 = []  # List of 2 lists, one for each band

        for s1_band in s1_bands:
            print(f"Processing Sentinel-1 band {s1_band}")
            s1_files = glob.glob(
                f"{s1_dir}/{yr}*/4326_{s1_band}.tif")

            # Sort the files by directory name that contains the date
            s1_files.sort(key=lambda x: x.split('/')[-2])

            all_s1.append(s1_files)

            print(
                f"Found {len(s1_files)} Sentinel-1 files for band {s1_band}")

        # [print(f"File: {f}") for f in s1_files]

        all_s1_files[str(yr)] = all_s1  # Store the list of files for each year

        all_s2 = []  # List of 12 lists, one for each band

        for s2_band in s2_bands:
            print(f"Processing Sentinel-2 band {s2_band}")
            s2_files = glob.glob(
                f"{s2_dir}/{yr}*/4326_{s2_band}.tif")

            # Sort the files by directory name that contains the date
            s2_files.sort(key=lambda x: x.split('/')[-2])

            all_s2_files.append(s2_files)

            print(
                f"Found {len(s2_files)} Sentinel-2 files for band {s2_band}")

        all_s2_files[str(yr)] = all_s2

        # [print(f"File: {f}") for f in s2_files]

    return all_s1_files, all_s2_files


# Iterate over each pixel location

@njit(parallel=True)
def parallel_impute(data, imputed_data):
    height, width = data.shape[1], data.shape[2]
    for i in prange(height):
        for j in range(width):
            time_series_data = data[:, i, j]
            imputed_values = impute_missing_values(time_series_data)
            imputed_data[:, i, j] = imputed_values


if __name__ == "__main__":

    s1_dir = "/work/mech-ai-scratch/rtali/gis-sentinel1/final_s1"
    s2_dir = "/work/mech-ai-scratch/rtali/gis-sentinel2/final_s2_v3"
    years = [2019, 2020, 2021, 2022, 2023]
    s1_bands = ["vv", "vh"]
    s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05',
                'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    # Obtain files for the specified years. It is a list of lists.
    s1_files, s2_files = obtain_files(
        [2019, 2020, 2021, 2022, 2023], s1_dir, s2_dir)

    # For each year, for each band, aggregate the files and save them.
    # At this point, 1 file per band per year is created.

    for yr in years:
        print(f"Processing year {yr}")
        # For each year, process the files for each band
        for s1_band_files in s1_files[str(yr)]:
            aggregate_bands_by_year(s1_band_files)

        for s2_band_files in s2_files[str(yr)]:
            aggregate_bands_by_year(s2_band_files)

    # perform time based imputation
    """
    We take the aligned_union_multiband.tif file and perform time based imputation on it.
    For every pixel, we take the corressponding values for all the bands and perform curve fitting.
    """

    # Read the aligned union multiband file

    for yr in years:
        print(f"Imputing year {yr}")

        for s1_band in s1_bands:

            with rasterio.open(f"./imputation_time_based/images_by_year/{yr}/aligned_union_multiband_{s1_band}.tif") as src:
                data = src.read()
                transform = src.transform
                crs = src.crs
                meta = src.meta
                height, width = data.shape[1], data.shape[2]
                print(f"Data shape: {data.shape}")

            # Create an output array to store the imputed data
            imputed_data = np.full(data.shape, np.nan, dtype=np.float32)

            # Perform parallel imputation
            parallel_impute(data, imputed_data)

            # Save the imputed data to a new GeoTIFF file
            output_path = f"./imputation_time_based/images_by_year/{yr}/{s1_band}_imputed.tif"
            with rasterio.open(output_path, "w", **meta) as dst:
                for i in range(data.shape[0]):
                    dst.write(imputed_data[i], i + 1)
                dst.transform = transform
                dst.crs = crs
                dst.nodata = np.nan
            print(f"Saved imputed {s1_band} image for year {yr}")

        for s2_band in s2_bands:
            with rasterio.open(f"./imputation_time_based/images_by_year/{yr}/aligned_union_multiband_{s2_band}.tif") as src:
                data = src.read()
                transform = src.transform
                crs = src.crs
                meta = src.meta
                height, width = data.shape[1], data.shape[2]
                print(f"Data shape: {data.shape}")

            # Create an output array to store the imputed data
            imputed_data = np.full(data.shape, np.nan, dtype=np.float32)

            # Perform parallel imputation
            parallel_impute(data, imputed_data)

            # Save the imputed data to a new GeoTIFF file
            output_path = f"./imputation_time_based/images_by_year/{yr}/{s2_band}_imputed.tif"
            with rasterio.open(output_path, "w", **meta) as dst:
                for i in range(data.shape[0]):
                    dst.write(imputed_data[i], i + 1)
                dst.transform = transform
                dst.crs = crs
                dst.nodata = np.nan
            print(f"Saved imputed {s2_band} image for year {yr}")
        print(f"Finished imputing year {yr}")
    print("All years processed successfully.")
