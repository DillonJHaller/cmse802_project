'''
This code takes in a shape file and pulls HLS date for the points to a dataframe.
'''
import numpy as np
import geopandas as gpd
import pandas as pd
import os
import rasterio

mosaic_directory = "D:\\HLS_Data\\Mosaics"

def pull_hls_data(shapefile_path, mosaic_directory):
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

    return results_df


#Pull training data
shapefile_train = "data\\Train_Test_Points\\training_points.shp"
train_df = pull_hls_data(shapefile_train, mosaic_directory)

#Pull testing data
shapefile_test = "data\\Train_Test_Points\\testing_points.shp"
test_df = pull_hls_data(shapefile_test, mosaic_directory)

##Further feature engineering:
def combine_sentinel_landsat(df, feature_pairs):
    '''
    Function to combine Sentinel and Landsat features in the dataframe by averaging comparable features.

    args:
        df: DataFrame containing HLS features
        feature_pairs: List of tuples, each containing the names of the Landsat and Sentinel features to be averaged.
    '''
    for landsat_feature, sentinel_feature in feature_pairs:
        if landsat_feature in df.columns and sentinel_feature in df.columns:
            combined_feature_name = f"Avg_{'_'.join(landsat_feature.split('_')[1:])}"  
            df[combined_feature_name] = df[[landsat_feature, sentinel_feature]].mean(axis=1)
            # Drop the original features
            df.drop(columns=[landsat_feature, sentinel_feature], inplace=True)

#List of feature pairs to combine
feature_pairs = [
    ('L30_2024_B01winter', 'S30_2024_B01winter'), #Aerosol
    ('L30_2024_B02winter', 'S30_2024_B02winter'), #Blue
    ('L30_2024_B03winter', 'S30_2024_B03winter'), #Green
    ('L30_2024_B04winter', 'S30_2024_B04winter'), #Red
    ('L30_2024_B05winter', 'S30_2024_B08winter'), #NIR
    ('L30_2024_B06winter', 'S30_2024_B11winter'), #SWIR1
    ('L30_2024_B07winter', 'S30_2024_B12winter'), #SWIR2
    ('L30_2024_EVI2winter', 'S30_2024_EVI2winter'), #EVI2

    ('L30_2024_B01summer', 'S30_2024_B01summer'),
    ('L30_2024_B02summer', 'S30_2024_B02summer'),
    ('L30_2024_B03summer', 'S30_2024_B03summer'),   
    ('L30_2024_B04summer', 'S30_2024_B04summer'),
    ('L30_2024_B05summer', 'S30_2024_B08summer'),
    ('L30_2024_B06summer', 'S30_2024_B11summer'),
    ('L30_2024_B07summer', 'S30_2024_B12summer'),
    ('L30_2024_EVI2summer', 'S30_2024_EVI2summer'),

    ('L30_2024_B01yearly', 'S30_2024_B01yearly'),
    ('L30_2024_B02yearly', 'S30_2024_B02yearly'),
    ('L30_2024_B03yearly', 'S30_2024_B03yearly'),
    ('L30_2024_B04yearly', 'S30_2024_B04yearly'),
    ('L30_2024_B05yearly', 'S30_2024_B08yearly'),
    ('L30_2024_B06yearly', 'S30_2024_B11yearly'),
    ('L30_2024_B07yearly', 'S30_2024_B12yearly')
]

for df in [train_df, test_df]:
    combine_sentinel_landsat(df, feature_pairs)

#Get summer/winter differences for selected bands and EVI2
def add_seasonal_differences(df, bands):
    '''
    Function to add seasonal difference features to the dataframe.

    args:
        df: DataFrame containing HLS features
        bands: List of band identifiers to compute seasonal differences for.
    '''
    for band in bands:
        winter_col = f"Avg_2024_{band}winter"
        summer_col = f"Avg_2024_{band}summer"
        if winter_col in df.columns and summer_col in df.columns:
            diff_col = f"Diff_{band}_summer_winter"
            df[diff_col] = df[summer_col] - df[winter_col]

bands_to_process = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'EVI2']
for df in [train_df, test_df]:
    add_seasonal_differences(df, bands_to_process)

#Save out dataframes
train_csv = "data\\Training_Data.csv"
train_df.to_csv(train_csv, index = False)
test_csv = "data\\Testing_Data.csv"
test_df.to_csv(test_csv, index = False)