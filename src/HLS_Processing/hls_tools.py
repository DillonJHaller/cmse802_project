'''
Contains functions used in processing HLS data
'''
import numpy as np
import os
import rasterio


#Landsat bands:
#2 - Blue
#3 - Green
#4 - Red
#5 - NIR
#6 - SWIR1
#7 - SWIR2
L_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
#Sentinel-2 bands:
#2 - Blue
#3 - Green
#4 - Red
#5 - Red Edge 1
#6 - Red Edge 2
#7 - Red Edge 3
#8A - NIR narrow
#11 - SWIR1
#12 - SWIR2
S_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
#See Ju et al., 2025 for more details.

# Should take an input directory which contains subdirectories for each date.
# Should save out individual band averages as GeoTIFFs in an output directory.
def process_directory(input_directory, band_list, output_directory):
    #Get important information out of the path
    parts = input_directory.split(os.sep)
    program = parts[-6]
    year = parts[-5]
    tile = f"{parts[-4]}{parts[-3]}{parts[-2]}{parts[-1]}"
    #One subdirectory for each date
    subdirectories = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
    #Traverse each band in turn
    for band in band_list:
        #Sum and count per pixel will be used for an average
        band_sum = None
        valid_pixel_count = None
        #Traverse each date subdirectory
        for subdirectory in subdirectories:
            band_file = os.path.join(input_directory, subdirectory, f"{subdirectory}.{band}.tif")
            #Mask file contains cloud/shadow/water info
            #See Ju et al., 2025 for more details on Fmask values
            mask_file = os.path.join(input_directory, subdirectory, f"{subdirectory}.Fmask.tif")
            if os.path.exists(band_file):
                with rasterio.open(band_file) as src:
                    band_data = src.read(1).astype(np.float32) #Convert to float so averages work properly
                    mask = rasterio.open(mask_file)
                    mask_data = mask.read(1)
                    #Taking first date to initialize arrays
                    if band_sum is None:
                        band_sum = np.zeros_like(band_data)
                        valid_pixel_count = np.zeros_like(band_data)
                    valid_mask = ((band_data != src.nodata) & #Valid pixels: not nodata and not cloud/shadow or nearby
                                   (mask_data % 8 == 0))
                    mask_data = None
                    mask.close()
                    band_sum[valid_mask] += band_data[valid_mask]
                    
                    valid_pixel_count[valid_mask] += 1
        #Only rights out if there is at least one good date
        if band_sum is not None and valid_pixel_count is not None:
            #Compute pixel-wise average
            band_average = np.divide(band_sum, valid_pixel_count, out=np.zeros_like(band_sum), where=valid_pixel_count!=0)
            output_path = os.path.join(output_directory, f"{program}_{year}_{tile}_{band}_average.tif")
            #Get file properties from the last of the input files
            with rasterio.open(band_file) as src:
                profile = src.profile
            #Update profile for single band float32 output
            profile.update(dtype=rasterio.float32, count=1, compress='lzw')
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(band_average.astype(rasterio.float32), 1)

    pass
