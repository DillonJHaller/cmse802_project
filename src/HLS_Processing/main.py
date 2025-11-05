'''
Main function to process HLS data
'''

import numpy as np
import os
import rasterio
import hls_tools as hls

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
for directory in L2024_directories:
    hls.process_directory(directory, hls.L_bands, output_directory)
for directory in S2024_directories:
    hls.process_directory(directory, hls.S_bands, output_directory)