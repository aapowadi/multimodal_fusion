import pystac_client
import planetary_computer
import geopandas as gpd
import requests
import os
from pathlib import Path

# Define output directory for GeoTIFF files
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

# Connect to Planetary Computer STAC API
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

# Define Iowa's bounding box (approximate, in decimal degrees)
# Iowa: -96.6397 to -90.1401 longitude, 40.3754 to 43.5011 latitude
bbox = [-96.6395, 40.3754, -90.1401, 43.5014]  # [minx, miny, maxx, maxy]

# Search for 3DEP DEMs (1-meter or 10-meter resolution)
collection = "3dep-seamless"  # 3DEP DEM dataset
search = catalog.search(
    collections=[collection],
    bbox=bbox,
    query={"gsd": {"eq": 1}},  # 1-meter resolution; change to 10 for 10-meter
)

# Retrieve items (datasets)
items = search.item_collection()
print(f"Found {len(items)} items matching the criteria.")

# Download GeoTIFF files
for item in items:
    # Get the DEM asset (GeoTIFF)
    asset = item.assets.get("data")  # The GeoTIFF file
    if not asset:
        print(f"No data asset found for item {item.id}")
        continue

    # Generate signed URL for download
    signed_href = planetary_computer.sign(asset).href

    # Extract filename from URL
    base_url = signed_href.split("?")[0]
    filename = Path(base_url).name
    output_path = os.path.join(output_dir, filename)

    # Download the GeoTIFF
    print(f"Downloading {filename}...")
    response = requests.get(signed_href)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Saved {filename}")
    else:
        print(f"Failed to download {filename}: HTTP {response.status_code}")

print("Download complete!")
