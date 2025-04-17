"""
Input : Sent-1 and Sent-2 images
Output : Imputed Sentinel-1/Sentinel-2 images saved as a .tif files. Same directory structure as input

Assumptions:
- We assume that pixels are spatially IID

Logic:

- For each year, for each band, for each pizel location, we pick all pixel values
- We fit a polynomial curve to the data points
- We predict the missing values using the fitted polynomial coefficients
- We save the imputed images in the same directory structure as the input images

"""

from logging_helper import setup_logging
import os
import numpy as np
import rasterio
from geotiff_operations import update_geotiff_pixel
from curve_fitting import curve_fitting, predict_missing_values
from glob import glob
import argparse
from numba import njit, prange
from tqdm import tqdm


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Impute missing values in Sentinel-1/Sentinel-2 images.")
    parser.add_argument("modality", type=str, help="S1 or S2")
    args = parser.parse_args()

    modality = args.modality
    if modality not in ["S1", "S2"]:
        raise ValueError("Modality must be either 'S1' or 'S2'.")

    # Set up logging
    myLogger = setup_logging(f"./imputation_time_based/{modality}.log")

    os.makedirs(f"./imputation_time_based/{modality}", exist_ok=True)

    for yr in [2019, 2020, 2021, 2022, 2023]:

        myLogger.info(f"Processing year {yr}")
