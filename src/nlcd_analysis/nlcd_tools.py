'''
This script recodes raw NLCD landcover data into simplified classes, simplified classes reflecting natural, agricultural, and developed land covers.
'''
import numpy as np
#Recode array to simpler classes
def reclass_lc(lc_array):
    '''
    Simple function to reclassify an NLCD array to a simpler format, with only four land cover classes (+ No Data)

    args:
        lc_array: An NLCD raster loaded in as a Numpy array

    References for NLCD value meanings:
        11: "#466b9f", #Open Water
        12: "#d1def8", #Snow/Ice
        21: "#dec5c5", #Dev Open space
        22: "#d99282", #Dev low intensity
        23: "#eb0000", #Dev med intensity
        24: "#ab0000", #Dev high intensity
        31: "#b3ac9f", #Barren
        41: "#68ab5f", #Deciduous
        42: "#1c5f2c", #Evergreen
        43: "#b5c58f", #Mixed
        52: "#ccb879", #Shrub/scrub
        71: "#dfdfc2", #Grassland/herbaceous
        81: "#dcd939", #Pasture/hay
        82: "#ab6c28", #Cultivated crops
        90: "#b8d9eb", #Woody wetlands
        95: "#6c9fb8"  #Herbaceous wetlands

    Reclassed values:
        0: NoData
        1: Pasture/hay
        2: Cultivated crops
        3: Non-agricultural/non-developed
        4: Developed land
    '''
    reclass = np.zeros_like(lc_array) #Create a blank array of same size as original, fill it with no data
    # np.isin() returns a boolean array which can be used as an index, reflecting the positions where the value in the lc_array is within the list of classes given
    reclass[np.isin(lc_array, [81])] = 1 #Where original has pasture/hay
    reclass[np.isin(lc_array, [82])] = 2 #Where original has cropland
    reclass[np.isin(lc_array, [11, 12, 31, 41, 42, 43, 52, 71, 90, 95])] = 3 #Where original has any other class (Should Open water be separated? tbd)
    reclass[np.isin(lc_array, [21, 22, 23, 24])] = 4 #Any developed class
    return reclass

#Convert to long-term pattern class as described in Martin et al., 2025
# Need the following classes:
########## Constants
#### 1: Stable pasture/hay
#### 2: Stable cropland
#### 3: Stable non-agricultural/non-developed
########## Transitions
#### 4: Transitioned from pasture/hay to cropland
#### 5: Transitioned from pasture to non-agricultural/non-developed
#### 6: Transitioned from cropland to pasture
#### 7: Transitioned from cropland to NAND
#### 8: Transitioned from NAND to pasture
#### 9: Transitioned from NAND to crops
########## Other
#### 0: NoData or erratic patterns (both will be left out of analysis)
#### 11: Developed land at any time

#Function to compute LTPCs
def ltpc_conversion(ts):
    '''
    Function to convert from a simplified land cover time series to an array of long-term pattern class values. Classification is as described
    in Martin et al., 2025, but expanded to encompass cropland, pasture, and non-ag as three separate possibilities, rather than simply ag/non-ag.

    args:
        ts: A time series taking the form of a 3-dimensional array

    Reclassed land cover values (input):
        0: NoData
        1: Pasture/hay
        2: Cultivated crops
        3: Non-agricultural/non-developed
        4: Developed land
    
    Output LTPC values:
        Constants
            1: Stable pasture/hay
            2: Stable cropland
            3: Stable non-agricultural/non-developed
        Transitions
            4: Transitioned from pasture/hay to cropland
            5: Transitioned from pasture to non-agricultural/non-developed
            6: Transitioned from cropland to pasture
            7: Transitioned from cropland to NAND
            8: Transitioned from NAND to pasture
            9: Transitioned from NAND to crops
        Other
            0: NoData or erratic patterns (both will be left out of analysis)
            11: Developed land at any time
    '''
    ltpc = np.zeros_like(ts[0], dtype=np.uint8)
    n_years = ts.shape[0]

    # Find stable classes
    for class_value, stable_code in [(1, 1), (2, 2), (3, 3)]:
        count_class = np.sum(ts == class_value, axis=0)
        is_stable = count_class >= (n_years - 1)
        ltpc[is_stable] = stable_code

    # Find transitions (Iterate across years to ensure only one change)
    for start_code, end_code, transition_code in [
        (1, 2, 4), (1, 3, 5),
        (2, 1, 6), (2, 3, 7),
        (3, 1, 8), (3, 2, 9)
    ]:
        transition_mask = (ts[0] == start_code) & (ts[-1] == end_code)
        trans_flag = np.zeros_like(transition_mask, dtype=bool)
        for year in range(1, n_years):
            transition_mask = transition_mask & ( #We have not already invalidated the transition
            ((ts[year] == start_code) & ~trans_flag) | #Still in start class, transition not yet occurred
            (ts[year] == end_code) #Transition has occurred, still in end class
            )
            trans_flag = trans_flag | (ts[year] == end_code) #Mark that transition has occurred
        ltpc[transition_mask] = transition_code

    # Find developed land (easy part)
    ltpc[np.any(ts == 4, axis=0)] = 11

    return ltpc
