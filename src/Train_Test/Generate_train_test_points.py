'''
This code generates shapefiles which contain points that will be used for pulling out training and testing data, no need to separate them later.
'''

import numpy as np
import geopandas as gpd
import rasterio #An alternative library using for raster data
from rasterio.transform import xy
from shapely.geometry import Point
import os

#Variables
n_train_points, n_test_points = 100, 25
raster_path = "D:\\NLCD\\LPTC\\LPTC.tif"
out_path = "data\\Train_Test_Points"
HLS_crs = "32615"  # UTM Zone 15N

#Load up the raster of long-term pattern classes
with rasterio.open(raster_path) as src:
    ltpc_array = src.read(1)  
    transform = src.transform
    crs = src.crs #Coordinate reference system

#Empty storage to generate sample points
training_geoms = []
training_labels = []
testing_geoms = []
testing_labels = []

#In this case, we are only interested in the 9 main LTPCs (1-9), excluding NoData (0) and Developed (11)
for val in range(1, 10):
    #Get the indices of the array where the value matches the current LTPC
    rows, cols = np.where(ltpc_array == val)

    if len(rows) < 1000:
        continue  # SKip this if there are not enough samples

    point_indices = np.random.choice(len(rows), size=n_train_points+n_test_points, replace=False)
    
    #Now split into training and testing
    training_point_indices = point_indices[:n_train_points]
    testing_point_indices = point_indices[n_train_points:]

    #Convert indices to UTM coordinates
    train_xs, train_ys, = xy(transform, rows[training_point_indices], cols[training_point_indices], offset='center')
    test_xs, test_ys, = xy(transform, rows[testing_point_indices], cols[testing_point_indices], offset='center')

    #Build out the points
    training_geoms.extend([Point(x, y) for x, y in zip(train_xs, train_ys)])
    training_labels.extend([val] * n_train_points)
    testing_geoms.extend([Point(x, y) for x, y in zip(test_xs, test_ys)])
    testing_labels.extend([val] * n_test_points)

#Now create the dataframes
train_gdf = gpd.GeoDataFrame({'geometry': training_geoms, 'LTPC': training_labels}, crs=crs)
test_gdf = gpd.GeoDataFrame({'geometry': testing_geoms, 'LTPC': testing_labels}, crs=crs)

#Reproject to HLS CRS
train_gdf = train_gdf.to_crs(epsg=HLS_crs)
test_gdf = test_gdf.to_crs(epsg=HLS_crs)

#Save out the files, shapefile, which is probably the most common format for storing geographic vector data
#If you are unfamiliar with shapefiles, note that they actually consist of multiple files with the same name but different extensions
train_gdf.to_file(os.path.join(out_path, "training_points.shp"))
test_gdf.to_file(os.path.join(out_path, "testing_points.shp"))