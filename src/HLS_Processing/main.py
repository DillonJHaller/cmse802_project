'''
Main script to process HLS data. Converts raw HLS data from the simple format into a format that can be used to pull values for the later machine learning algorithm.
'''

import numpy as np
import os
import rasterio
import hls_tools as hls
from itertools import product

#Process the naive averages of bands
base_output_directory = "D:\\HLS_Data\\Processed\\"
#Create output directory if it doesn't exist
if not os.path.exists(base_output_directory):
    os.makedirs(base_output_directory)

#Go through and process all directories for Landsat and Sentinel-2 in turn
for directory, band in product(hls.Landsat_directories, hls.L_bands):
    #Get program, year, tile info from directory path
    parts = directory.split(os.sep)
    program = parts[-6]
    year = parts[-5]
    tile = f"{parts[-4]}{parts[-3]}{parts[-2]}{parts[-1]}"

    for season, dates in [("winter", ("335", "059")), ("summer", ("152", "243")), ("yearly", ("001", "366"))]:
        season_output_dir = os.path.join(base_output_directory, f"{season.capitalize()}_Averages")
        if not os.path.exists(season_output_dir):
            os.makedirs(season_output_dir)
        
        band_average, profile = hls.create_band_average(directory, band, dates=dates)
        if band_average is not None:
            #Write out GeoTIFF
            output_file = os.path.join(season_output_dir, f"{program}_{year}_{band}{season}_{tile}.tif")
            profile.update(nodata=-999) #Set nodata value
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(band_average.astype(rasterio.float32), 1)
        else:
            #Print warning if no data found
            print(f"Warning: No valid data found for {directory}, band {band}, season {season}")

        
for directory, band in product(hls.Sentinel_directories, hls.S_bands):
    parts = directory.split(os.sep)
    program = parts[-6]
    year = parts[-5]
    tile = f"{parts[-4]}{parts[-3]}{parts[-2]}{parts[-1]}"
    
    for season, dates in [("winter", ("335", "059")), ("summer", ("152", "243")), ("yearly", ("001", "366"))]:
        season_output_dir = os.path.join(base_output_directory, f"{season.capitalize()}_Averages")
        if not os.path.exists(season_output_dir):
            os.makedirs(season_output_dir)
        
        band_average, profile = hls.create_band_average(directory, band, dates=dates)
        if band_average is not None:
            #Write out GeoTIFF
            output_file = os.path.join(season_output_dir, f"{program}_{year}_{band}{season}_{tile}.tif")
            profile.update(nodata=-999) #Set nodata value
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(band_average.astype(rasterio.float32), 1)
        else:
            #Print warning if no data found
            print(f"Warning: No valid data found for {directory}, band {band}, season {season}")


############
# Mosaics #
############


#Get all tifs within base processed data folder
processed_folder = "D:\\HLS_Data\\Processed\\"
mosaic_folder = "D:\\HLS_Data\\Mosaics\\"

#Create mosaic folder if it doesn't exist
if not os.path.exists(mosaic_folder):
    os.makedirs(mosaic_folder)

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

############################
# Band index calculations #
############################

base_output_directory = "D:\\HLS_Data\\Mosaics"
#Create output directory if it doesn't exist
if not os.path.exists(base_output_directory):
    os.makedirs(base_output_directory)

#Compute EVI2 mosaics
for program, season in product(["L30", "S30"], ["winter", "summer", "yearly"]):
    if program == "L30":
        red_band_file = os.path.join(mosaic_folder, f"L30_2024_B04{season}_mosaic.tif")
        nir_band_file = os.path.join(mosaic_folder, f"L30_2024_B05{season}_mosaic.tif")
    else:
        red_band_file = os.path.join(mosaic_folder, f"S30_2024_B04{season}_mosaic.tif")
        nir_band_file = os.path.join(mosaic_folder, f"S30_2024_B08{season}_mosaic.tif")
    
    with rasterio.open(red_band_file) as red_src:
        red_band = red_src.read(1)
        profile = red_src.profile
    with rasterio.open(nir_band_file) as nir_src:
        nir_band = nir_src.read(1)
    
    evi2_array = hls.computeEVI2(nir_band, red_band)
    output_file = os.path.join(base_output_directory, f"{program}_2024_EVI2{season}.tif")
    
    profile.update(nodata=-999) #Set nodata value
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(evi2_array.astype(rasterio.float32), 1)

