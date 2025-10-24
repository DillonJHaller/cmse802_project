Repository for Dillon Haller's CMSE 802 semester project

<h3> Project Title: Using machine learning to identify former farmland based on current characteristics.</h3>

<strong>Brief Description:</strong> Land that was formerly under cultivation may have distinct soil, vegetation, etc. characteristics from that which was not recently under cultivation. I will attempt to demonstrate that there are such differences in my study area of central Missouri by training a machine learning model on recent remotely sensed data over both former farmland and non-farmland. A machine-learning model which succesfully makes the distinction in the validation dataset could indicate that there are landscape patterns on former farmland which are visible from space.

<h3> Project Objectives: </h3>

1. I will use the National Land Cover Database to identify former agricultural land. This involves loading in the original dataset as a series of NumPy arrays or a higher-dimensional array and making comparisons across time between single locations during different years. I anticipate completing this within the first two weeks.
2. I will train a machine learning model to detect a difference between former agricultural land and non-agricultural land. This will involve recent satellite data and labels derived from step 1. I expect to use a random forest model but may switch to others as I learn more. I hope to complete this around the end of October, though I will likely then spend extensive time tweaking the model until the end of November.
3. I will validate the aforementioned model. I will accomplish this by holding back 20% of the original label dataset and comparing those labels to the ones predicted by the model. Then I will compute a confusion matrix. I expect to complete this shortly after the first run of the machine learning model, and will then use those results to improve the model.

<h3> Instructions for running code: </h3>

There's no code yet, so I'll add this later

<h3> Required packages and dependencies: </h3>

This code is entirely written in Python. You will need Python 3.x to run it, as well as the following packages:
1. NumPy
2. Pandas
3. MatPlotLib
4. GDAL - The Geospatial Data Abstraction Library
5. Rasterio - A wrapper for many GDAL functions that is somewhat easier to work with
6. Geopandas - An extension to Pandas that allows for columns to contain information on geographic locations
7. Shapely

<h3> Data: </h3>

The following datasets were used and are too large to be stored on GitHub
1. National Land Cover Database
2. Harmonized Landsat Sentinel-2
