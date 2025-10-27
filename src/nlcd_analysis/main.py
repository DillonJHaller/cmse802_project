'''
Main function. Runs the process to generate labels from NLCD from start to finish
'''
import numpy as np
import pandas as pd
import rasterio
import os
import nlcd_tools

#Set up constants
raster_directory = "D:\\NLCD\\Masked"
reclassed_directory = "D:\\NLCD\\Reclassed"
LTPC_directory = "D:\\NLCD\\LTPC"
start_year = 1985

###Reclassifying NLCD data to a simpler scheme
#Read in original NLCD file locations
raster_files = [f for f in os.listdir(raster_directory) if f.lower().endswith('.tif')]

#Get rasterio info from first file
with rasterio.open(raster_files[0]) as src:
    profile = src.profile

#Loop through all years, making reclassifications
for i, raster_file in enumerate(raster_files):
    year = start_year + i
    with rasterio.open(os.path.join(raster_directory, raster_file)) as dst:
        lc_array = dst.read(1)
        rc_array = nlcd_tools.reclass_lc(lc_array)
    with rasterio.open(os.path.join(reclassed_directory, f"RC_{year}"), 'w', **profile) as dst:
        dst.write(rc_array.astype(rasterio.int16), 1)

###Generate LTPCs from reclassed NLCD
#Read in names of reclassed NLCD files
raster_files = [f for f in os.listdir(reclassed_directory) if f.lower().endswith('.tif')]

#Get 3D array of reclassed data (For now, only last ten years) (2014-2023)
time_series = []
for raster_file in raster_files[-10:]:
    with rasterio.open(os.path.join(raster_directory, raster_file)) as dst:
        lc_array = dst.read(1)
        time_series.append(lc_array)
time_series = np.array(time_series)

#Compute and write out LTPC raster
ltpc = nlcd_tools.ltpc_conversion(time_series)
with rasterio.open(os.path.join(LTPC_directory, "LTPC"), 'w', **profile) as dst:
    dst.write(ltpc.astype(rasterio.int16), 1)



