'''
Main script to process HLS data. Converts raw HLS data from the simple format into a format that can be used to pull values for the later machine learning algorithm.
'''

import numpy as np
import os
import rasterio
import hls_tools as hls
from itertools import product

#Process the naive averages of bands
output_directory = "D:\\HLS_Data\\Processed\\Naive_Averages"

#Go through and process all directories for Landsat and Sentinel-2 in turn
for directory, band in product(hls.Landsat_directories, hls.L_bands):
    band_average, profile = hls.create_band_average(directory, band)
    if band_average is not None:
        #Write out GeoTIFF
        parts = directory.split(os.sep)
        program = parts[-6]
        year = parts[-5]
        tile = f"{parts[-4]}{parts[-3]}{parts[-2]}{parts[-1]}"
        output_file = os.path.join(output_directory, f"HLS_{program}_{year}_{band}naverage_{tile}.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(band_average.astype(rasterio.float32), 1)
        
for directory, band in product(hls.Sentinel_directories, hls.S_bands):
    band_average, profile = hls.create_band_average(directory, band)
    if band_average is not None:
        #Write out GeoTIFF
        parts = directory.split(os.sep)
        program = parts[-6]
        year = parts[-5]
        tile = f"{parts[-4]}{parts[-3]}{parts[-2]}{parts[-1]}"
        output_file = os.path.join(output_directory, f"HLS_{program}_{year}_{band}naverage_{tile}.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(band_average.astype(rasterio.float32), 1)

#Get all tifs within base processed data folder
processed_folder = "D:\\HLS_Data\\Processed\\"
mosaic_folder = "D:\\HLS_Data\\Mosaics\\"
#Get all tifs, make sure to check subdirectories if needed
tif_files = []
for root, dirs, files in os.walk(processed_folder):
    for file in files:
        if file.endswith(".tif"):
            tif_files.append(os.path.join(root, file))

#Mosaic groups of four together
#Get the unique parts before tile number
metrics = list(set(['_'.join(f.split('_')[:-1]) for f in tif_files]))
for m in metrics:
    #Get all files matching this metric
    matching_files = [f for f in tif_files if f.startswith(m)]
    hls.mosaic_tifs(matching_files, mosaic_folder)
