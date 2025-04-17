import os
import csv
import re


def get_dates(directory):
    """
    Returns a set of dates that exist as folders (in date format) within the given directory.
    """
    return {d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))}

def get_weather_dates_by_filename(weather_dir):
    """
    Walk through each variable sub‑folder under weather_dir,
    extract any YYYYMMDD or YYYY-MM-DD date strings from filenames,
    and return as a set of 'YYYY-MM-DD'.
    """
    date_regex = re.compile(r'(\d{4})[-–]?(\d{2})[-–]?(\d{2})')
    dates = set()

    for var in os.listdir(weather_dir):
        var_path = os.path.join(weather_dir, var)
        if not os.path.isdir(var_path):
            continue
        for fname in os.listdir(var_path):
            m = date_regex.search(fname)
            if m:
                year, month, day = m.groups()
                dates.add(f"{year}-{month}-{day}")
    return dates

def main():
    # Directory paths where each modality's data is stored
    sentinel1_dir = '/work/mech-ai-scratch/rtali/gis-sentinel1/final_s1'
    sentinel2_dir = '/work/mech-ai-scratch/rtali/gis-sentinel2/final_s2_v3'
    modis_dir     = '/work/mech-ai-scratch/rtali/gis-modis/modis'
    weather_dir   = '/work/mech-ai-scratch/rtali/AI_READY_IOWA/WEATHER_TIFFS'
    
    # Get date sets for each modality
    dates_s1     = get_dates(sentinel1_dir)
    dates_s2     = get_dates(sentinel2_dir)
    dates_modis  = get_dates(modis_dir)
    dates_weather = get_weather_dates_by_filename(weather_dir)

    # Compute the intersection (common dates) across all modalities
    common_dates = (
        dates_s1
        & dates_s2
        & dates_modis
        & dates_weather
    )

    # Print to console
    print("Common overlapping dates across Sentinel‑1, Sentinel‑2, MODIS, and Weather:")
    for date in sorted(common_dates):
        print(date)

    # Save to CSV
    csv_filename = "common_dates.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["date"])
        for date in sorted(common_dates):
            writer.writerow([date])

    print(f"\n→ Saved {len(common_dates)} dates to '{csv_filename}'.")

if __name__ == "__main__":
    main()
