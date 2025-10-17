import numpy as np
import pandas as pd
import csv
import os
from osgeo import gdal

#Set up constants
raster_directory = "D:\\NLCD\\Reclassed"

#Read in names of reclassed NLCD files
raster_files = [f for f in os.listdir(raster_directory) if f.lower().endswith('.tif')]

time_series = []

#Get 3D array of reclassed data
for raster_file in raster_files:
    with gdal.Open(os.path.join(raster_directory, raster_file)) as dataset:
        band = dataset.GetRasterBand(1)
        lc_array = band.ReadAsArray()
        time_series.append(lc_array)

time_series = np.array(time_series)