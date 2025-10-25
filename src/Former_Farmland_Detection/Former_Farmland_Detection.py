#This script processes the last few years of reclassified NLCD data to generate a Long-Term Pattern Class (LTPC) raster as described in Martin et al., 2025.

import numpy as np
import pandas as pd
import csv
import os
from osgeo import gdal
gdal.UseExceptions()

#Reference for value meanings:
#0: NoData
#1: Pasture/hay
#2: Cultivated crops
#3: Non-agricultural/non-developed
#4: Developed land

#Set up constants
raster_directory = "D:\\NLCD\\Reclassed"

#Read in names of reclassed NLCD files
raster_files = [f for f in os.listdir(raster_directory) if f.lower().endswith('.tif')]

#Get basic GDAL information from the first tif (This is a bit of spaghetti, I'll clean it up later)
r1 = raster_files[0]
with gdal.Open(os.path.join(raster_directory, r1)) as dataset:
    gt = dataset.GetGeoTransform()
    prj = dataset.GetProjection()
    dr = gdal.GetDriverByName('GTiff')
    rx = dataset.RasterXSize
    ry = dataset.RasterYSize
    band = dataset.GetRasterBand(1)
    dt = band.DataType

#Get 3D array of reclassed data (For now, only last ten years) (2014-2023)
time_series = []
for raster_file in raster_files[-10:]:
    with gdal.Open(os.path.join(raster_directory, raster_file)) as dataset:
        band = dataset.GetRasterBand(1)
        lc_array = band.ReadAsArray()
        time_series.append(lc_array)

time_series = np.array(time_series)

#Convert to long-term pattern class as described in Martin et al., 2025
# Need the following classes:
########## Constants
#### 1: Stable pasture/hay
#### 2: Stable cropland
#### 3: Stable non-agricultural/non-developed
########## Transitions
#### 4: Transitioned from pasture/hay to cropland
#### 5: Transitioned from pasture to non-agricultural/non-developed
#### 6: Transitioned from cropland to pasture
#### 7: Transitioned from cropland to NAND
#### 8: Transitioned from NAND to pasture
#### 9: Transitioned from NAND to crops
########## Other
#### 0: NoData or erratic patterns (both will be left out of analysis)
#### 11: Developed land at any time

#Function to compute LTPCs
def ltpc_conversion(ts):
    ltpc = np.zeros_like(ts[0], dtype=np.uint8)
    n_years = ts.shape[0]

    # Find stable classes
    for class_value, stable_code in [(1, 1), (2, 2), (3, 3)]:
        count_class = np.sum(ts == class_value, axis=0)
        is_stable = count_class >= (n_years - 1)
        ltpc[is_stable] = stable_code

    # Find transitions (Iterate across years to ensure only one change)
    for start_code, end_code, transition_code in [
        (1, 2, 4), (1, 3, 5),
        (2, 1, 6), (2, 3, 7),
        (3, 1, 8), (3, 2, 9)
    ]:
        transition_mask = (ts[0] == start_code) & (ts[-1] == end_code)
        trans_flag = np.zeros_like(transition_mask, dtype=bool)
        for year in range(1, n_years):
            transition_mask = transition_mask & ( #We have not already invalidated the transition
            ((ts[year] == start_code) & ~trans_flag) | #Still in start class, transition not yet occurred
            (ts[year] == end_code) #Transition has occurred, still in end class
            )
            trans_flag = trans_flag | (ts[year] == end_code) #Mark that transition has occurred
        ltpc[transition_mask] = transition_code

    # Find developed land (easy part)
    ltpc[np.any(ts == 4, axis=0)] = 11

    return ltpc

lptc = ltpc_conversion(time_series)
#Write out LTPC raster

out_ds = dr.Create(f'D:\\NLCD\\LPTC\\LPTC.tif', rx, ry, 1, gdal.GDT_Int8) 
out_ds.SetGeoTransform(gt)
out_ds.SetProjection(prj)
out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(lptc)
out_band.FlushCache()
out_ds = None