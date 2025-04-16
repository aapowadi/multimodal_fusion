import rasterio
import numpy as np


def update_geotiff_pixel(geotiff_path, row, col, new_value):
    # Open the GeoTIFF file in read-write mode
    with rasterio.open(geotiff_path, 'r+') as dataset:
        # Read the data for the first band (modify as needed for multi-band)
        data = dataset.read(1)

        # Check if the specified pixel is within bounds
        if row < 0 or row >= dataset.height or col < 0 or col >= dataset.width:
            raise ValueError("Specified pixel coordinates are out of bounds")

        # Update the pixel value
        data[row, col] = new_value

        # Write the updated data back to the first band
        dataset.write(data, 1)
