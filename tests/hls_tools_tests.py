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
        
