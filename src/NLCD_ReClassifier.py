import numpy as np
import pandas as pd
from osgeo import gdal
gdal.UseExceptions()
import os
import matplotlib.pyplot as plt
import csv

#Functions of analysis
##Should be able to take in tifs with georeferencing information and return a tif with georeferencing information
##Find all areas with constant cropland or pasture/hay landcover for five years, that subsequently have a non-agricultural, non developed landcover for five years
##Specifically pick out contiguous clusters of such areas larger than a size threshold
##Identify the date of the transition

#Recode array to simpler classes
def reclass_lc(lc_array):
    reclass = np.zeros_like(lc_array) #Create a blank array of same size as original, fill it with no data
    # np.isin() returns a boolean array which can be used as an index, reflecting the positions where the value in the lc_array is within the list of classes given
    reclass[np.isin(lc_array, [81])] = 1 #Where original has pasture/hay
    reclass[np.isin(lc_array, [82])] = 2 #Where original has cropland
    reclass[np.isin(lc_array, [11, 12, 31, 41, 42, 43, 52, 71, 90, 95])] = 3 #Where original has any other class (Should Open water be separated? tbd)
    reclass[np.isin(lc_array, [21, 22, 23, 24])] = 4 #Any developed class
    return reclass

#Set up constants
raster_directory = "D:\\NLCD\\Masked"
nlcd_vals = (11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95)
nlcd_colorscheme = {
    11: "#466b9f", #Open Water
    12: "#d1def8", #Snow/Ice
    21: "#dec5c5", #Dev Open space
    22: "#d99282", #Dev low intensity
    23: "#eb0000", #Dev med intensity
    24: "#ab0000", #Dev high intensity
    31: "#b3ac9f", #Barren
    41: "#68ab5f", #Deciduous
    42: "#1c5f2c", #Evergreen
    43: "#b5c58f", #Mixed
    52: "#ccb879", #Shrub/scrub
    71: "#dfdfc2", #Grassland/herbaceous
    81: "#dcd939", #Pasture/hay
    82: "#ab6c28", #Cultivated crops
    90: "#b8d9eb", #Woody wetlands
    95: "#6c9fb8"  #Herbaceous wetlands
}

#Read in NLCD file locations
raster_files = [f for f in os.listdir(raster_directory) if f.lower().endswith('.tif')]
print(raster_files)


#Get basic GDAL information from the first tif
r1 = raster_files[0]
with gdal.Open(os.path.join(raster_directory, r1)) as dataset:
    gt = dataset.GetGeoTransform()
    prj = dataset.GetProjection()
    dr = gdal.GetDriverByName('GTiff')
    rx = dataset.RasterXSize
    ry = dataset.RasterYSize
    band = dataset.GetRasterBand(1)
    dt = band.DataType

#Loop through all years
for i, raster_file in enumerate(raster_files):
    year = 1985 + i
    with gdal.Open(os.path.join(raster_directory, raster_file)) as dataset:
        band = dataset.GetRasterBand(1)
        lc_array = band.ReadAsArray()
        rc_array = reclass_lc(lc_array)

        out_ds = dr.Create(f'D:\\NLCD\\Reclassed\\RC_{year}.tif', rx, ry, 1, gdal.GDT_UInt16) #Storing externally due to file sizes
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(prj)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(rc_array)
        out_band.FlushCache()
        out_ds = None