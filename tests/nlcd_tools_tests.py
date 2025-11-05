'''
This script performs unit tests to check the functionality of the functions contained within 'nlcd_tools.py'
'''
import unittest
#Import functions from nlcd_tools
from src.nlcd_analysis.nlcd_tools import reclass_lc
from src.nlcd_analysis.nlcd_tools import ltpc_conversion
import numpy as np

#Test methods
class TestNLCDTools(unittest.TestCase):

    def test_reclass_lc_input(self):
        '''
        Test the reclass_lc function
        '''
        #First, function should reject any input that is not a two-dimensional numpy array
        with self.assertRaises(TypeError): #Copilot suggested this structure with the 'with' statement. It may or may not be standard, but it seems more readable.
            reclass_lc([[11, 21], [31, 41]]) #List input
        with self.assertRaises(TypeError):
            reclass_lc("twelve")
        with self.assertRaises(ValueError):
            reclass_lc(np.array([2,3,5]))
        with self.assertRaises(ValueError):
            reclass_lc(np.array([[[1,1],[2,2]], [[3,3],[4,4]]]))
        
    def test_reclass_lc_output(self):
        '''
        Test the output of the reclass_lc function
        '''
        #Function should return an array of dimension equal to the input array
        test_array = np.array([[11,11],[31,31]])
        self.assertEqual(reclass_lc(test_array).shape, test_array.shape)

        #On a known input, function should return expected output
        test_array = np.array([[11,21,81],[82,41,95],[23,24,12]])
        expected_output = np.array([[3,4,1],[2,3,3],[4,4,3]])
        np.testing.assert_array_equal(reclass_lc(test_array), expected_output)
    
    def test_ltpc_conversion_input(self):
        '''
        Test the ltpc_conversion function
        '''
        #First, function should reject any input that is not a three-dimensional numpy array
        with self.assertRaises(TypeError):
            ltpc_conversion([[[1,2],[3,4]], [[1,2],[3,4]]]) #List input
        with self.assertRaises(TypeError):
            ltpc_conversion("thirteen")
        with self.assertRaises(ValueError):
            ltpc_conversion(np.array([[1,2],[3,4]]))
        with self.assertRaises(ValueError): #Too many dimensions
            ltpc_conversion(np.array([[[[1,3,4],[2,3,4]], [[1,3,4],[2,3,4]],
                                       [[1,3,4],[2,3,4]], [[1,3,4],[2,3,4]]],
                                      [[[1,3,4],[2,3,4]], [[1,3,4],[2,3,4]],
                                       [[1,3,4],[2,3,4]], [[1,3,4],[2,3,4]]]]))
    
    def test_ltpc_conversion_output(self):
        '''
        Test the output of the ltpc_conversion function
        '''
        #Function should return an array of dimension equal to the last two dimensions of the input array
        test_array = np.array([[[1,1],[1,1]], [[1,1],[1,1]], [[1,1],[1,1]]])
        self.assertEqual(ltpc_conversion(test_array).shape, test_array.shape[1:3])

        #On a known input, function should return expected output
        test_array = np.array([
            [[1,1,3],[2,2,3],[3,3,3]],
            [[1,1,3],[2,2,3],[3,3,3]],
            [[2,2,3],[2,2,3],[3,3,4]],
            [[2,2,3],[3,3,3],[3,3,4]],
            [[2,2,3],[3,3,3],[3,3,4]]
        ])
        expected_output = np.array([[4,4,3],[7,7,3],[3,3,11]])
        np.testing.assert_array_equal(ltpc_conversion(test_array), expected_output)

