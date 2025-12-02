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

##Throw out missing values
train_df.replace(-999, np.nan, inplace=True)
test_df.replace(-999, np.nan, inplace=True)

#Find any rows with highly implausible values (<5 or > 5) and set to NaN
for col in train_df.columns:
    if col not in ['ID', 'Northing', 'Easting', 'LTPC']:
        train_df.loc[(train_df[col] < -5) | (train_df[col] > 5), col] = np.nan
        test_df.loc[(test_df[col] < -5) | (test_df[col] > 5), col] = np.nan


##Further feature engineering:
def combine_sentinel_landsat(df, feature_pairs):
    '''
    Function to combine Sentinel and Landsat features in the dataframe by averaging comparable features.

    args:
        df: DataFrame containing HLS features
        feature_pairs: List of tuples, each containing the names of the Landsat and Sentinel features to be averaged, and the overall feature name.
    '''
    for landsat_feature, sentinel_feature, combined_feature_name in feature_pairs:
        if landsat_feature in df.columns and sentinel_feature in df.columns:
            df[combined_feature_name] = df[[landsat_feature, sentinel_feature]].mean(axis=1)
            # Drop the original features
            df.drop(columns=[landsat_feature, sentinel_feature], inplace=True)

#List of feature pairs to combine
feature_pairs = [
    ('L30_2024_B01winter', 'S30_2024_B01winter', '2024_Aerosolwinter'),
    ('L30_2024_B02winter', 'S30_2024_B02winter', '2024_Bluewinter'),
    ('L30_2024_B03winter', 'S30_2024_B03winter', '2024_Greenwinter'),
    ('L30_2024_B04winter', 'S30_2024_B04winter', '2024_Redwinter'),
    ('L30_2024_B05winter', 'S30_2024_B08winter', '2024_NIRwinter'),
    ('L30_2024_B06winter', 'S30_2024_B11winter', '2024_SWIR1winter'),
    ('L30_2024_B07winter', 'S30_2024_B12winter', '2024_SWIR2winter'),
    ('L30_2024_EVI2winter', 'S30_2024_EVI2winter', '2024_EVI2winter'),

    ('L30_2024_B01summer', 'S30_2024_B01summer', '2024_Aerosolsummer'),
    ('L30_2024_B02summer', 'S30_2024_B02summer', '2024_Bluesummer'),
    ('L30_2024_B03summer', 'S30_2024_B03summer', '2024_Greensummer'),
    ('L30_2024_B04summer', 'S30_2024_B04summer', '2024_Redsummer'),
    ('L30_2024_B05summer', 'S30_2024_B08summer', '2024_NIRsummer'),
    ('L30_2024_B06summer', 'S30_2024_B11summer', '2024_SWIR1summer'),
    ('L30_2024_B07summer', 'S30_2024_B12summer', '2024_SWIR2summer'),
    ('L30_2024_EVI2summer', 'S30_2024_EVI2summer', '2024_EVI2summer'),

    ('L30_2024_B01yearly', 'S30_2024_B01yearly', '2024_Aerosolyearly'),
    ('L30_2024_B02yearly', 'S30_2024_B02yearly', '2024_Blueyearly'),
    ('L30_2024_B03yearly', 'S30_2024_B03yearly', '2024_Greenyearly'),
    ('L30_2024_B04yearly', 'S30_2024_B04yearly', '2024_Redyearly'),
    ('L30_2024_B05yearly', 'S30_2024_B08yearly', '2024_NIRyearly'),
    ('L30_2024_B06yearly', 'S30_2024_B11yearly', '2024_SWIR1yearly'),
    ('L30_2024_B07yearly', 'S30_2024_B12yearly', '2024_SWIR2yearly'),
    ('L30_2024_EVI2yearly', 'S30_2024_EVI2yearly', '2024_EVI2yearly')
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
        winter_col = f"2024_{band}winter"
        summer_col = f"2024_{band}summer"
        if winter_col in df.columns and summer_col in df.columns:
            diff_col = f"Diff_{band}_summer_winter"
            df[diff_col] = df[summer_col] - df[winter_col]

bands_to_process = ['Aerosol', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'EVI2']
for df in [train_df, test_df]:
    add_seasonal_differences(df, bands_to_process)

#Save out dataframes
train_csv = "data\\Training_Data.csv"
train_df.to_csv(train_csv, index = False)
test_csv = "data\\Testing_Data.csv"
test_df.to_csv(test_csv, index = False)