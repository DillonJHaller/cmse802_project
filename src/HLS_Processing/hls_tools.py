'''
Contains functions used in processing HLS data
Band number reference:
Landsat bands:
02 - Blue
03 - Green
04 - Red
05 - NIR
06 - SWIR1
07 - SWIR2
Sentinel-2 bands:
02 - Blue
03 - Green
04 - Red
05 - Red Edge 1
06 - Red Edge 2
07 - Red Edge 3
8A - NIR narrow
11 - SWIR1
12 - SWIR2
#See Ju et al., 2025 for more details.
'''
import numpy as np
import os
import rasterio

L_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
S_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']


# Should take an input directory which contains subdirectories for each date.
def create_band_average(input_directory, band):
    '''
    Function to create a numpy array which has the average reflectance values for a single band

    args:
        input_directory: Directory which contains the raw data. 
        band: String which identifies which band is being averaged.
    '''
    #Reject if input directories do not exist
    if not os.path.isdir(input_directory):
        raise ValueError("Input directory does not exist")
    
    #One subdirectory for each date
    subdirectories = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
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

    #Only returns if there is at least one good date
    if band_sum is not None and valid_pixel_count is not None:
        #Compute pixel-wise average
        band_average = np.divide(band_sum, valid_pixel_count, out=np.zeros_like(band_sum), where=valid_pixel_count!=0)

        #Return all info needed to write out GeoTIFF
        #Get profile from one of the input files
        sample_file = os.path.join(input_directory, subdirectories[0], f"{subdirectories[0]}.{band}.tif")
        with rasterio.open(sample_file) as src:
            profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw') #Update profile for output
        return band_average, profile
    else:
        return None, None


    def mosaic_tifs(tif_list, out_folder = None):
        '''
        Function to mosaic a set of four tifs together, given paths to them. If given an out folder, save out the mosaic
        '''
        pass

