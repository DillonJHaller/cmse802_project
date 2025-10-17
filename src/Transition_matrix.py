#######################################################################################################################
# This is an EDA script which creates matrices that chart the transitions from any land cover to any land cover in any 
# set of years. Currently it checks the transitions that occur across the full series and also every pair of years since 2014.
# This information is used for some simple visualizations via an ipynb found in the 'notebooks' directory.
####################################################################################################################### 
# Keeping this here because I use it for reference
# #nlcd_colorscheme = {
#     11: "#466b9f", #Open Water
#     12: "#d1def8", #Snow/Ice
#     21: "#dec5c5", #Dev Open space
#     22: "#d99282", #Dev low intensity
#     23: "#eb0000", #Dev med intensity
#     24: "#ab0000", #Dev high intensity
#     31: "#b3ac9f", #Barren
#     41: "#68ab5f", #Deciduous
#     42: "#1c5f2c", #Evergreen
#     43: "#b5c58f", #Mixed
#     52: "#ccb879", #Shrub/scrub
#     71: "#dfdfc2", #Grassland/herbaceous
#     81: "#dcd939", #Pasture/hay
#     82: "#ab6c28", #Cultivated crops
#     90: "#b8d9eb", #Woody wetlands
#     95: "#6c9fb8"  #Herbaceous wetlands
# }

#Import necessary libraries
import os
import numpy as np
import pandas as pd
from osgeo import gdal
gdal.UseExceptions()
from itertools import product
import csv

#Set up constants
raster_directory = "D:\\NLCD\\Masked"
nlcd_vals = (11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95)

#Read in NLCD arrays
raster_files = [f for f in os.listdir(raster_directory) if f.lower().endswith('.tif')]

#Empty time series
time_series = []

#Get basic GDAL information from the first tif
# r1 = raster_files[0]
# with gdal.Open(os.path.join(raster_directory, r1)) as dataset:
#     gt = dataset.GetGeoTransform()
#     prj = dataset.GetProjection()
#     dr = gdal.GetDriverByName('GTiff')
#     rx = dataset.RasterXSize
#     ry = dataset.RasterYSize
#     band = dataset.GetRasterBand(1)
#     dt = band.DataType
#Holding on to this for now to copy/paste elsewhere

#Loop through all years
for raster_file in raster_files:
    with gdal.Open(os.path.join(raster_directory, raster_file)) as dataset:
        band = dataset.GetRasterBand(1)
        lc_array = band.ReadAsArray()
        time_series.append(lc_array)

#Convert time series to 3D array
time_series = np.array(time_series)

#Function to detect all transitions and save out a matrix as a csv
def transition_matrix(time_series, nlcd_vals, start_year = 1985, end_year = 2023):
    
    #Set up the matrix as a Pandas dataframe (This makes it easier to name the rows and columns)
    full_ts_trans_matrix = np.zeros((len(nlcd_vals), len(nlcd_vals)), dtype = np.float64)
    full_ts_trans_df = pd.DataFrame(full_ts_trans_matrix, columns = nlcd_vals)
    full_ts_trans_df.index = nlcd_vals

    start_index = start_year - 1985
    end_index = end_year - 1985

    #Flatten out beginning and end images
    start = time_series[start_index].flatten()
    end = time_series[end_index].flatten()

    #Get rid of NoData values
    valid = (start != 0) & (end != 0) #'&' Broadcasts the logical operation to create a boolean array
    start = start[valid] #Take out all invalid pixels
    end = end[valid]

    #Total number of valid pixels
    tot = len(start)

    from collections import defaultdict #A type of dictionary that automatically initializes a default value at a non-existent key if you try to adjust the value there
    transition_counts = defaultdict(int)

    #Count instances of all possible transitions
    for s, e in zip(start, end):
        transition_counts[(s, e)] += 1

    #Put that into a transition matrix
    for (s, e), count in transition_counts.items():
        full_ts_trans_df.loc[s, e] = count/tot * 100

    #Write to csv
    full_ts_trans_df.to_csv(f'results\\{start_year}_to_{end_year}_Trans.csv')
    return full_ts_trans_df

transition_matrix(time_series, nlcd_vals)

for year in range(2014, 2022):
    transition_matrix(time_series, nlcd_vals, start_year = year, end_year = year + 1)


