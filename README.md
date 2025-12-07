Repository for Dillon Haller's CMSE 802 semester project

<h3> Project Title: Using machine learning to identify former farmland based on current characteristics.</h3>

<strong>Brief Description:</strong> Land that was formerly under cultivation may have distinct soil, vegetation, etc. characteristics from that which was not recently under cultivation. I will attempt to demonstrate that there are such differences in my study area of central Missouri by training a machine learning model on recent remotely sensed data over both former farmland and non-farmland. A machine-learning model which succesfully makes the distinction in the validation dataset could indicate that there are landscape patterns on former farmland which are visible from space.

<h3> Project Objectives: </h3>

1. I will use the National Land Cover Database to identify former agricultural land. This involves loading in the original dataset as a series of NumPy arrays or a higher-dimensional array and making comparisons across time between single locations during different years. I anticipate completing this within the first two weeks.
2. I will train a machine learning model to detect a difference between former agricultural land and non-agricultural land. This will involve recent satellite data and labels derived from step 1. I expect to use a random forest model but may switch to others as I learn more. I hope to complete this around the end of October, though I will likely then spend extensive time tweaking the model until the end of November.
3. I will validate the aforementioned model. I will accomplish this by holding back 20% of the original label dataset and comparing those labels to the ones predicted by the model. Then I will compute a confusion matrix. I expect to complete this shortly after the first run of the machine learning model, and will then use those results to improve the model.

<h3> Instructions for running code: </h3>

**NOTE:** This process depends heavily on very large files (>1 GB each, and very numerous). Those files cannot be saved on GitHub. These instructions assume you have the data saved locally. If you do not, most of the code under `src` will not run.

<h4> 1a. Pre-processing land cover data </h4>

The scripts for pre-processing land cover data are found under `src\nlcd_analysis.` In brief, running `main.py` will save out landcover datasets that have been reclassified into only four land cover types, pasture, cropland, non-agricultural/non-developed (NAND), and developed. This script will then fetch those reclassified landcover datasets and convert them to Long-term pattern classes (LTPCs), which track the trajectories of the land cover over the last ten years of the original dataset. We are interested in pasture, farmland, and NAND areas and any area which transitioned between any two of those. Any other patterns are not considered further. These LTPCs constitute the labels for the machine learning model.

`nlcd_tools.py` contains the functions that are used to pre-process land cover data. These are `reclass_lc`, which converts the original land cover classes into the four classes of interest, and `ltpc_conversion` which takes several years of reclassified land cover data and creates the LTPC array therefrom.

<h4> 1b. Pre-processing HLS data </h4>

The scripts for pre-processing the HLS satellite data are found under `src\HLS_Processing`. This step contains most of the feature engineering. Running `main.py` will convert raw satellite observations into summarized layers from which values can be extracted. `hls_tools.py` contains the functions which are used to summarize the HLS data, as well as a few constants that are fed in. `create_band_average` averages values across a given set of dates, ignoring those that have poor values in the QA band. `mosaic_tifs` is used to combine the four tiles of the study area together, and `ComputeEVI2` computes the EVI2 index. Note that `ComputeNDVI` is not currently used. 

<strong> WARNING: this step is very RAM and CPU intensive </strong>. It may take several hours to run locally.

<h4> 2. Generating the training and testing datasets </h4>

The scripts under `src\Train_Test` actually generate the data science-friendly datasets. NLCD and HLS processing must be done before proceeding to this step. `Generate_train_test_points.py` creates shapefiles at random pixels in each LTPC. These shapefiles are saved in the repo. `Pull_HLS_Data.py` uses the Rasterio library to grab values from the HLS data at each of the previously generated points. It also performs a bit of additional feature engineering that can be done more quickly without generating whole raster mosaics.

<h4> 3. Model Implementation </h4>

The scripts under `src\Model_Implementation` perform all work related to actually implementing the model. This is the only step which can be done without access to the original remote sensing archive. `run_model.py` creates, trains, and evaluates a random forest and support vector classifier separately. `model_analysis.py` reads in those models to perform a bit of basic analysis on which features are important to each model. Similarly to other folders within `src`, `modeling_tools.py` is a module containing functions used in model generation and evaluation.

<h4> Exploratory Analysis </h4>

In addition to the main workflow, there are a few notebooks contained under `explo` which perform some optional exploratory data analysis. The script and notebook under `transitions` generate some basic time series from the NLCD data that track changes in the prevalence of specific land cover categories over time. `reflectance_values.ipynb` takes the spreadsheets generated by `Pull_HLS_Data` and creates a series of bar charts showing reflectance values for each of the LTPCs.

<h3> Required packages and dependencies: </h3>

This code is entirely written in Python. You will need Python 3.x to run it, as well as the following packages:
1. NumPy
2. Pandas
3. MatPlotLib
4. scikit-learn
5. GDAL - The Geospatial Data Abstraction Library
6. Rasterio - A wrapper for many GDAL functions that is somewhat easier to work with
7. Geopandas - An extension to Pandas that allows for columns to contain information on geographic locations
8. Shapely
9. pickle
10. unittest - Only if you want to run the unittests yourself

<h3> Data: </h3>

The following datasets were used and are too large to be stored on GitHub
1. National Land Cover Database
2. Harmonized Landsat Sentinel-2

See more at `Data_Statement.md` under the `data` folder
