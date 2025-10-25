Repository for Dillon Haller's CMSE 802 semester project

<h3> Project Title: Using machine learning to identify former farmland based on current characteristics.</h3>

<strong>Brief Description:</strong> Land that was formerly under cultivation may have distinct soil, vegetation, etc. characteristics from that which was not recently under cultivation. I will attempt to demonstrate that there are such differences in my study area of central Missouri by training a machine learning model on recent remotely sensed data over both former farmland and non-farmland. A machine-learning model which succesfully makes the distinction in the validation dataset could indicate that there are landscape patterns on former farmland which are visible from space.

<h3> Project Objectives: </h3>

1. I will use the National Land Cover Database to identify former agricultural land. This involves loading in the original dataset as a series of NumPy arrays or a higher-dimensional array and making comparisons across time between single locations during different years. I anticipate completing this within the first two weeks.
2. I will train a machine learning model to detect a difference between former agricultural land and non-agricultural land. This will involve recent satellite data and labels derived from step 1. I expect to use a random forest model but may switch to others as I learn more. I hope to complete this around the end of October, though I will likely then spend extensive time tweaking the model until the end of November.
3. I will validate the aforementioned model. I will accomplish this by holding back 20% of the original label dataset and comparing those labels to the ones predicted by the model. Then I will compute a confusion matrix. I expect to complete this shortly after the first run of the machine learning model, and will then use those results to improve the model.

<h3> Instructions for running code: </h3>

**NOTE:** This process depends heavily on very large files (>1 GB each, and very numerous). Those files cannot be saved on GitHub. These instructions assume you have the data saved locally. If you do not, most of the code under `src` will not run.

Most of the code so far written is pre-processing. There are three pre-processing steps, each with their own folder. The NLCD processing and HLS processing need to be done first, but between the two of them can be done in either order. After that, the training and testing data need to be generated. Finally, the actual model implementation is done at the end.

<h4> 1a. Pre-processing land cover data </h4>

The scripts for pre-processing land cover data are found under `src\Former_Farmland_Detection.` In brief, running `NLCD_ReClassifier.py` will save out landcover datasets that have been reclassified into only four land cover types, pasture, cropland, non-agricultural/non-developed (NAND), and developed. Running `Former_Farmland_Detection.py` will fetch those reclassified landcover datasets and convert them to Long-term pattern classes (LTPCs), which track the trajectories of the land cover over the last ten years of the original dataset. We are interested in pasture, farmland, and NAND areas and any area which transitioned between any two of those. Any other patterns are not considered further. These LTPCs constitute the labels for the machine learning model.

`Transition_matrix.py` is also located here. It is not part of the model generation process, but used to look at more specific dates and types of land cover transitions.

<h4> 1b. Pre-processing HLS data </h4>

The scripts for pre-processing the HLS satellite data are found under `src\HLS_Processing`. This is the "Feature engineering" step. Currently, I have only been able to put up `Naive_HLS_Processing.py`, which computes average reflectance values over 2024. These aren't really very useful. Further HLS processing will be added here. Note that there is a step I completed outside this repo because it can be more easily done outside of Python for small datasets. This step is mosaicking the data (stitching tiles together). Note that even while not implemented all the way, this step is very RAM and processor intensive

<h4> 2. Generating the training and testing datasets </h4>

The scripts under `src\Train_Test` actually generate the data science-friendly datasets. NLCD and HLS processing must be done before proceeding to this step. `Generate_train_test_points.py` creates shapefiles at random pixels in each LTPC. These shapefiles are saved in the repo. `Pull_HLS_Data.py` uses the Rasterio library to grab values from the HLS data at each of the previously generated points.

<h4> 3. Model Implementation </h4>

The scripts under `src\Model_Implementation` will perform all work related to actually implementing the model. This is the only step which can be done without access to the original remote sensing archive. Currently, this only contains `Model_Implementation.py`, which creates a simple random forest model trained on the data pulled from HLS and the labels pulled from the NLCD.

<h3> Required packages and dependencies: </h3>

This code is entirely written in Python. You will need Python 3.x to run it, as well as the following packages:
1. NumPy
2. Pandas
3. MatPlotLib
4. GDAL - The Geospatial Data Abstraction Library
5. Rasterio - A wrapper for many GDAL functions that is somewhat easier to work with
6. Geopandas - An extension to Pandas that allows for columns to contain information on geographic locations
7. Shapely
8. pickle

<h3> Data: </h3>

The following datasets were used and are too large to be stored on GitHub
1. National Land Cover Database
2. Harmonized Landsat Sentinel-2

See more at `Data_Statement.md` under the `data` folder
