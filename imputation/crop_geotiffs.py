"""
Input : MODIS, Sent-1 and Sent-2 Geotiffs
Output : Cropped Geotiffs of size 64 X 64
"""

from splitraster import geo

def split_geotiff(input_file, output_folder, tile_size=64):
    """
    Split a large GeoTIFF file into smaller tiles using the splitraster package.

    Args:
        input_file (str): Path to the input GeoTIFF file.
        output_folder (str): Path to the folder where tiles will be saved.
        tile_size (int): Size of each tile (default is 64x64 pixels).
    """
    repetition_rate = 0  # No overlap between tiles
    overwrite = True     # Overwrite existing files if necessary
    
    # Split the image into tiles
    n_tiles = geo.split_image(
        input_file, 
        output_folder, 
        crop_size=tile_size, 
        repetition_rate=repetition_rate, 
        overwrite=overwrite
    )
    
    print(f"{n_tiles} tiles created from {input_file} and saved to {output_folder}")


if __name__ == "__main__":
    pass
