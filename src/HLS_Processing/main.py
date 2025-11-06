'''
Main script to process HLS data. Converts raw HLS data from the simple format into a format that can be used to pull values for the later machine learning algorithm.
'''

import numpy as np
import os
import rasterio
import hls_tools as hls
from itertools import product

#List out storage locations
L2024_directories = ["D:\\HLS_Data\\Data\\L30\\2024\\15\\S\\V\\D",
                     "D:\\HLS_Data\\Data\\L30\\2024\\15\\S\\W\\D",
                     "D:\\HLS_Data\\Data\\L30\\2024\\15\\S\\W\\C",
                     "D:\\HLS_Data\\Data\\L30\\2024\\15\\S\\X\\C"]
S2024_directories = ["D:\\HLS_Data\\Data\\S30\\2024\\15\\S\\V\\D",
                     "D:\\HLS_Data\\Data\\S30\\2024\\15\\S\\W\\D",
                     "D:\\HLS_Data\\Data\\S30\\2024\\15\\S\\W\\C",
                     "D:\\HLS_Data\\Data\\S30\\2024\\15\\S\\X\\C"]

output_directory = "D:\\HLS_Data\\Processed\\Naive_Averages"

#Go through and process all directories for Landsat and Sentinel-2 in turn
for directory, band in product(L2024_directories, hls.L_bands):
    band_average, profile = hls.create_band_average(directory, band)
    if band_average is not None:
        #Write out GeoTIFF
        parts = directory.split(os.sep)
        program = parts[-6]
        year = parts[-5]
        tile = f"{parts[-4]}{parts[-3]}{parts[-2]}{parts[-1]}"
        output_file = os.path.join(output_directory, f"HLS_{program}_{tile}_{year}_{band}_Naive_Average.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(band_average.astype(rasterio.float32), 1)
        
for directory, band in product(S2024_directories, hls.S_bands):
    band_average, profile = hls.create_band_average(directory, band)
    if band_average is not None:
        #Write out GeoTIFF
        parts = directory.split(os.sep)
        program = parts[-6]
        year = parts[-5]
        tile = f"{parts[-4]}{parts[-3]}{parts[-2]}{parts[-1]}"
        output_file = os.path.join(output_directory, f"HLS_{program}_{tile}_{year}_{band}_Naive_Average.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(band_average.astype(rasterio.float32), 1)