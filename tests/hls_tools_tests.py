'''
This script performs unit tests to check the results of the functions within 'hls_tools.py'
'''
import numpy as np
import os
import rasterio
from src.HLS_Processing.hls_tools import create_band_average
import unittest

#Test methods
class TestHLSTools(unittest.TestCase):

    def test_create_band_average_inputs(self):
        '''
        Test inputs to band average.
        '''
        #Reject an invalid input directory
        with self.assertRaises(ValueError):
            create_band_average("invalid_input_directory", 'B02')
    
    def test_create_band_average_functionality(self):
        '''
        Test functionality of band average function with synthetic data.
        (Copilot wrote most of this, but I fixed it up)
        '''
        #Create a temporary directory structure with synthetic data
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp()
        try:
            #Create subdirectories and synthetic data
            dates = ['20240101', '20240201']
            for i, date in enumerate(dates):
                date_dir = os.path.join(temp_dir, date)
                os.makedirs(date_dir)
                #Create synthetic band data
                band_data = np.array([[100+2*i, 150+2*i], [200+2*i, 250+2*i]], dtype=np.uint16)
                band_file = os.path.join(date_dir, f"{date}.B02.tif")
                with rasterio.open(
                    band_file, 'w',
                    driver='GTiff',
                    height=band_data.shape[0],
                    width=band_data.shape[1],
                    count=1,
                    dtype=band_data.dtype
                ) as dst:
                    dst.write(band_data, 1)
                #Create synthetic mask data (all valid)
                mask_data = np.array([[16, 16], [16, 16]], dtype=np.uint8)
                mask_file = os.path.join(date_dir, f"{date}.Fmask.tif")
                with rasterio.open(
                    mask_file, 'w',
                    driver='GTiff',
                    height=mask_data.shape[0],
                    width=mask_data.shape[1],
                    count=1,
                    dtype=mask_data.dtype
                ) as dst:
                    dst.write(mask_data, 1)
            #Run the band average function
            band_average, profile = create_band_average(temp_dir, 'B02')
            #Check that the average is correct
            expected_average = np.array([[101.0, 151.0], [201.0, 251.0]], dtype=np.float32)
            np.testing.assert_array_equal(band_average, expected_average)
        finally:
            #Clean up temporary directory
            shutil.rmtree(temp_dir)

    def test_mosaic_tifs_inputs(self):
        '''
        Test inputs to mosaic tifs function.
        '''
        from src.HLS_Processing.hls_tools import mosaic_tifs
        #Reject non-list input
        with self.assertRaises(TypeError):
            mosaic_tifs("not_a_list", out_folder=None)
        #Reject list with wrong number of files
        with self.assertRaises(ValueError):
            mosaic_tifs(["file1.tif", "file2.tif"], out_folder=None)
        #Reject non-existent output folder
        with self.assertRaises(ValueError):
            mosaic_tifs(["file1.tif", "file2.tif", "file3.tif", "file4.tif"], out_folder="non_existent_folder")
