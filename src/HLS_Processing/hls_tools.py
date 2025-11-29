'''
Contains functions used in processing HLS data
Band number reference:
Landsat bands:
01 - Coastal/Aerosol
02 - Blue
03 - Green
04 - Red
05 - NIR
06 - SWIR1
07 - SWIR2
09 - Cirrus
    (The thermal bands are held out)
10 - TIRS1 (Thermal infrared)
11 - TIRS2

Sentinel-2 bands:
01 - Coastal aerosol
02 - Blue
03 - Green
04 - Red
05 - Red Edge 1
06 - Red Edge 2
07 - Red Edge 3
08 - NIR broad
8A - NIR narrow
09 - Water vapor
10 - Cirrus
11 - SWIR1
12 - SWIR2
#See Ju et al., 2025 for more details.
'''
import numpy as np
import os
import rasterio
from rasterio.merge import merge

L_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B09']
S_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

#Scale factor for reflectance values
scale_factor = 10000.0

#Storage_Locations
Landsat_directories =  ["D:\\HLS_Data\\Data\\L30\\2024\\15\\S\\V\\D",
                        "D:\\HLS_Data\\Data\\L30\\2024\\15\\S\\W\\D",
                        "D:\\HLS_Data\\Data\\L30\\2024\\15\\S\\W\\C",
                        "D:\\HLS_Data\\Data\\L30\\2024\\15\\S\\X\\C"]
Sentinel_directories = ["D:\\HLS_Data\\Data\\S30\\2024\\15\\S\\V\\D",
                        "D:\\HLS_Data\\Data\\S30\\2024\\15\\S\\W\\D",
                        "D:\\HLS_Data\\Data\\S30\\2024\\15\\S\\W\\C",
                        "D:\\HLS_Data\\Data\\S30\\2024\\15\\S\\X\\C"]


# Should take an input directory which contains subdirectories for each date.
def create_band_average(input_directory, band, dates = ("001", "366")):
    '''
    Function to create a numpy array which has the average reflectance values for a single band

    args:
        input_directory: Directory which contains the raw data. 
        band: String which identifies which band is being averaged.
        dates: Tuple of strings indicating the start and end Julian dates to consider (inclusive)
    '''
    #Reject if input directories do not exist
    if not os.path.isdir(input_directory):
        raise ValueError("Input directory does not exist")
    #Reject invalid band inputs
    if band not in L_bands and band not in S_bands:
        raise ValueError("Invalid band specified")
    #Reject invalid date inputs
    if (not isinstance(dates, tuple) or len(dates) != 2 or
        not all(isinstance(d, str) for d in dates) or
        not all(d.isdigit() and len(d) == 3 for d in dates) or 
        (dates[0] < "001" or dates[0] > "366" or dates[1] < "001" or dates[1] > "366")):
        raise ValueError("Dates must be a tuple of two strings indicating start and end Julian dates")
    
    #One subdirectory for each date
    subdirectories = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
    #Sum and count per pixel will be used for an average
    band_sum = None
    valid_pixel_count = None
    #Traverse each date subdirectory
    for subdirectory in subdirectories:
        #Fetch date from subdirectory name
        date_string = subdirectory.split('.')[3]
        #Date string is in format: YYYYJJJ'T'HHMMSS
        julian_date = date_string[4:7]
        #Check if date is within range
        if dates[0] <= dates[1]:
            if julian_date < dates[0] or julian_date > dates[1]:
                continue
        else:
            if julian_date < dates[0] and julian_date > dates[1]:
                continue

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
                valid_mask = ((band_data != src.nodata) & #Valid pixels: not nodata and not cloud/shadow or nearby or ice
                            (mask_data % 16 == 0) &
                            (mask_data < 192)) #Aerosol optical thickness level less than "high"
                mask_data = None
                mask.close()
                band_sum[valid_mask] += band_data[valid_mask]
                
                valid_pixel_count[valid_mask] += 1

    #Only returns if there is at least one good date
    if band_sum is not None and valid_pixel_count is not None:
        #Compute pixel-wise average
        band_average = np.divide(band_sum, valid_pixel_count, out=np.full(band_sum.shape, -999*scale_factor), where=valid_pixel_count!=0)
        reflectances = band_average / scale_factor  #Scale to reflectance values

        #Return all info needed to write out GeoTIFF
        #Get profile from one of the input files
        sample_file = os.path.join(input_directory, subdirectories[0], f"{subdirectories[0]}.{band}.tif")
        with rasterio.open(sample_file) as src:
            profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw') #Update profile for output
        return reflectances, profile
    else:
        return None, None


def mosaic_tifs(tif_list, out_folder = None):
    '''
    Function to mosaic a set of four tifs together, given paths to them. If given an out folder, save out the mosaic

    args:
        tif_list: List of file locations of four tifs
        out_folder: Location to save out the mosaicked raster. If None, do not save the raster
    '''
    #Reject invalid inputs
    if not isinstance(tif_list, list):
        raise TypeError("tif_list must be a list of four tif file paths")
    if len(tif_list) != 4:
        raise ValueError("tif_list must contain exactly four tif file paths")
    if out_folder is not None and not os.path.isdir(out_folder):
        raise ValueError("Output folder does not exist")

    src_files = [rasterio.open(tif) for tif in tif_list]
    mosaic_array, mosaic_transform = merge(src_files)

    # Normalize mosaic output to a single 2D band
    mosaic_array = np.asarray(mosaic_array)
    if mosaic_array.ndim == 4:
        # e.g., (1,1,height,width) -> take the first band
        mosaic_array = mosaic_array[0, 0, :, :]
    elif mosaic_array.ndim == 3:
        # (bands, height, width) -> take the first band
        mosaic_array = mosaic_array[0, :, :]
    elif mosaic_array.ndim == 2:
        # already (height, width)
        pass
    else:
        for src in src_files:
            src.close()
        raise ValueError(f"Unexpected mosaic array shape: {mosaic_array.shape}")

    #Get profile from one of the input files
    profile = src_files[0].profile
    profile.update(dtype=rasterio.float32,
                   height=mosaic_array.shape[0],
                   width=mosaic_array.shape[1],
                   transform=mosaic_transform,
                   count=1,
                   compress='lzw',
                   nodata = -999)

    #Create output file name
    if out_folder is not None:
        parts = os.path.basename(tif_list[0]).split('_')
        metric = '_'.join(parts[:-1])
        output_file = os.path.join(out_folder, f"{metric}.tif")
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(mosaic_array.astype(rasterio.float32), 1)
    
    #Close all opened files
    for src in src_files:
        src.close()

def computeNDVI(nir_band, red_band):
    '''
    Function to compute NDVI from NIR and Red bands
    NDVI is an index which indicates vegetation health

    args:
        nir_band: Numpy array of NIR band reflectance values
        red_band: Numpy array of Red band reflectance values
    '''
    #Reject invalid inputs
    if not isinstance(nir_band, np.ndarray) or not isinstance(red_band, np.ndarray):
        raise TypeError("Input bands must be numpy arrays")
    if nir_band.shape != red_band.shape:
        raise ValueError("Input bands must have the same shape")

    ndvi = ((nir_band - red_band) / (nir_band + red_band))
    return ndvi

def computeEVI2(nir_band, red_band):
    '''
    Function to compute EVI from NIR, Red, and Blue bands
    EVI is an index which indicates vegetation health, optimized for high biomass regions

    args:
        nir_band: Numpy array of NIR band reflectance values
        red_band: Numpy array of Red band reflectance values

    Citation:
        Jiang, Z., Huete, A. R., Didan, K., & Miura, T. (2008). Development of a two-band enhanced vegetation index without a blue band. Remote Sensing of Environment, 112(10), 3833–3845. https://doi.org/10.1016/j.rse.2008.06.006

    '''
    #Reject invalid inputs
    if not isinstance(nir_band, np.ndarray) or not isinstance(red_band, np.ndarray):
        raise TypeError("Input bands must be numpy arrays")
    if nir_band.shape != red_band.shape:
        raise ValueError("Input bands must have the same shape")

    G = 2.5
    C = 2.4
    L = 1.0

    evi2 = G * (nir_band - red_band) / (nir_band + C * red_band + L)
    return evi2