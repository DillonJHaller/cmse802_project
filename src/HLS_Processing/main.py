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
    band_average, profile = hls.create_band_average(directory, band)
    if band_average is not None:
        #Write out GeoTIFF
        output_file = os.path.join(base_output_directory, "Naive_Averages", f"HLS_{program}_{year}_{band}naverage_{tile}.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(band_average.astype(rasterio.float32), 1)
    else:
        #Print warning if no data found
        print(f"Warning: No valid data found for {directory}, band {band}")

    #Winter average (Roughly Dec 1 to Feb 28)
    winter_average, profile = hls.create_band_average(directory, band, dates=("335", "059"))
    if winter_average is not None:
        #Write out GeoTIFF
        output_file = os.path.join(base_output_directory, "Winter_Averages", f"HLS_{program}_{year}_{band}winteraverage_{tile}.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(winter_average.astype(rasterio.float32), 1)
    else:
        #Print warning if no data found
        print(f"Warning: No valid data found for {directory}, band {band}, winter averages")

    
    #Summer average (Roughly Jun 1 to Aug 31)
    summer_average, profile = hls.create_band_average(directory, band, dates=("152", "243"))
    if summer_average is not None:
        #Write out GeoTIFF
        output_file = os.path.join(base_output_directory, "Summer_Averages", f"HLS_{program}_{year}_{band}summeraverage_{tile}.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(summer_average.astype(rasterio.float32), 1)
    else:
        #Print warning if no data found
        print(f"Warning: No valid data found for {directory}, band {band}, summer averages")

        
for directory, band in product(hls.Sentinel_directories, hls.S_bands):
    parts = directory.split(os.sep)
    program = parts[-6]
    year = parts[-5]
    tile = f"{parts[-4]}{parts[-3]}{parts[-2]}{parts[-1]}"
    band_average, profile = hls.create_band_average(directory, band)
    if band_average is not None:
        #Write out GeoTIFF
        output_file = os.path.join(base_output_directory, "Naive_Averages", f"HLS_{program}_{year}_{band}naverage_{tile}.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(band_average.astype(rasterio.float32), 1)
    else:
        #Print warning if no data found
        print(f"Warning: No valid data found for {directory}, band {band}")

    
    #Winter average (Roughly Dec 1 to Feb 28)
    winter_average, profile = hls.create_band_average(directory, band, dates=("335", "059"))
    if winter_average is not None:
        #Write out GeoTIFF
        output_file = os.path.join(base_output_directory, "Winter_Averages", f"HLS_{program}_{year}_{band}winteraverage_{tile}.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(winter_average.astype(rasterio.float32), 1)
    else:
        #Print warning if no data found
        print(f"Warning: No valid data found for {directory}, band {band}, winter averages")


    #Summer average (Roughly Jun 1 to Aug 31)
    summer_average, profile = hls.create_band_average(directory, band, dates=("152", "243"))
    if summer_average is not None:
        #Write out GeoTIFF
        output_file = os.path.join(base_output_directory, "Summer_Averages", f"HLS_{program}_{year}_{band}summeraverage_{tile}.tif")
        profile.update(nodata=0) #Set nodata value
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(summer_average.astype(rasterio.float32), 1)
    else:
        #Print warning if no data found
        print(f"Warning: No valid data found for {directory}, band {band}, summer averages")


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

base_output_directory = "D:\\HLS_Data\\Indices"
#Create output directory if it doesn't exist
if not os.path.exists(base_output_directory):
    os.makedirs(base_output_directory)

#Winter EVI2 mosaics
winter_red_landsat = os.path.join(mosaic_folder, "HLS_Landsat_2024_B04winteraverage_mosaic.tif")
winter_nir_landsat = os.path.join(mosaic_folder, "HLS_Landsat_2024_B05winteraverage_mosaic.tif")
winter_evi2_landsat = hls.computeEVI2(winter_nir_landsat, winter_red_landsat)
output_file = os.path.join(base_output_directory, "HLS_Landsat_2024_EVI2winter.tif")
profile = winter_evi2_landsat[1]
profile.update(nodata=0) #Set nodata value
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(winter_evi2_landsat[0].astype(rasterio.float32), 1)

winter_red_sentinel = os.path.join(mosaic_folder, "HLS_Sentinel-2_2024_B04winteraverage_mosaic.tif")
winter_nir_sentinel = os.path.join(mosaic_folder, "HLS_Sentinel-2_2024_B08winteraverage_mosaic.tif")
winter_evi2_sentinel = hls.computeEVI2(winter_nir_sentinel, winter_red_sentinel)
output_file = os.path.join(base_output_directory, "HLS_Sentinel-2_2024_EVI2winter.tif")
profile = winter_evi2_sentinel[1]   
profile.update(nodata=0) #Set nodata value
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(winter_evi2_sentinel[0].astype(rasterio.float32), 1)

#Summer EVI2 mosaics
summer_red_landsat = os.path.join(mosaic_folder, "HLS_Landsat_2024_B04summeraverage_mosaic.tif")
summer_nir_landsat = os.path.join(mosaic_folder, "HLS_Landsat_2024_B05summeraverage_mosaic.tif")
summer_evi2_landsat = hls.computeEVI2(summer_nir_landsat, summer_red_landsat)
output_file = os.path.join(base_output_directory, "HLS_Landsat_2024_EVI2summer.tif")
profile = summer_evi2_landsat[1]
profile.update(nodata=0) #Set nodata value
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(summer_evi2_landsat[0].astype(rasterio.float32), 1)

summer_red_sentinel = os.path.join(mosaic_folder, "HLS_Sentinel-2_2024_B04summeraverage_mosaic.tif")
summer_nir_sentinel = os.path.join(mosaic_folder, "HLS_Sentinel-2_2024_B08summeraverage_mosaic.tif")
summer_evi2_sentinel = hls.computeEVI2(summer_nir_sentinel, summer_red_sentinel)
output_file = os.path.join(base_output_directory, "HLS_Sentinel-2_2024_EVI2summer.tif")
profile = summer_evi2_sentinel[1]   
profile.update(nodata=0) #Set nodata value
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(summer_evi2_sentinel[0].astype(rasterio.float32), 1)