'''
This script recodes raw NLCD landcover data into simplified classes, simplified classes reflecting natural, agricultural, and developed land covers.
'''

import numpy as np
import pandas as pd
from osgeo import gdal
gdal.UseExceptions()
import os
import matplotlib.pyplot as plt
import csv

#For reference:
# nlcd_colorscheme = {
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

#Recode array to simpler classes
def reclass_lc(lc_array):
    reclass = np.zeros_like(lc_array) #Create a blank array of same size as original, fill it with no data
    # np.isin() returns a boolean array which can be used as an index, reflecting the positions where the value in the lc_array is within the list of classes given
    reclass[np.isin(lc_array, [81])] = 1 #Where original has pasture/hay
    reclass[np.isin(lc_array, [82])] = 2 #Where original has cropland
    reclass[np.isin(lc_array, [11, 12, 31, 41, 42, 43, 52, 71, 90, 95])] = 3 #Where original has any other class (Should Open water be separated? tbd)
    reclass[np.isin(lc_array, [21, 22, 23, 24])] = 4 #Any developed class
    return reclass


