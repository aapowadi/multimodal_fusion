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

