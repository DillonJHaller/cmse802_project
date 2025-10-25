#This code takes in a shape file and pulls HLS date for the points to a dataframe.

import numpy as np
import geopandas as gpd
import pandas as pd
import os
import rasterio

mosaic_directory = "D:\\HLS_Data\\Processed\\Naive_Averages\\Mosaicked"

def pull_hls_data(shapefile_path, mosaic_directory, output_csv):
    #Read in the shapefile
    gdf = gpd.read_file(shapefile_path)
    #Prepare a dataframe to hold results
    results_df = pd.DataFrame()
    results_df['ID'] = range(len(gdf))
    results_df['LTPC'] = gdf['LTPC'] #This is the label column
    results_df['Northing'] = gdf.geometry.y
    results_df['Easting'] = gdf.geometry.x

    #List all mosaic files
    mosaic_files = [f for f in os.listdir(mosaic_directory) if f.endswith('.tif')]
    
    #For each mosaic file, extract values at point locations
    for mosaic_file in mosaic_files:
        feature_name = mosaic_file.split('.')[-2]  # Extract feature name from filename
        feature_values = []
        with rasterio.open(os.path.join(mosaic_directory, mosaic_file)) as src:
            for point in gdf.geometry:
                for val in src.sample([(point.x, point.y)]):
                    feature_values.append(val[0])
        results_df[feature_name] = feature_values

    #Save the results to CSV
    results_df.to_csv(output_csv, index=False)

#Pull training data
shapefile_path = "data\\Train_Test_Points\\training_points.shp"
output_csv = "data\\Training_Data.csv"
pull_hls_data(shapefile_path, mosaic_directory, output_csv)

#Pull testing data
shapefile_path = "data\\Train_Test_Points\\testing_points.shp"
output_csv = "data\\Testing_Data.csv"
pull_hls_data(shapefile_path, mosaic_directory, output_csv)