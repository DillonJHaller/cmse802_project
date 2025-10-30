<h3> Former Farmland Detection</h3>

This section of code processes raw NLCD data to generate labels. `main.py` runs the workflow for my directory setup, `nlcd_tools.py` contains the functions
There are two functions here which are used for the pre-processing of land cover data, in the given order:

1. `reclass_lc` is used to simplify the NLCD land cover data into just five classes, three of which are of interest
--pasture/hay (1)
--cultivated/crops (2)
--non-agricultural/non-developed (3)

2. `ltpc_conversion` is used to scan through the most recent ten years of land cover data and identify nine trajectories of interest, which will form the labels for the much more limited dataset used in the machine learning model. The nine trajectories are all possible combinations of starting and ending with one of the three aforementioned classes - giving three stable trajectories and six change trajectories. These are computed as an extension of the method used by Martin et al., 2025, in which stable trajectories must have at least 9/10 years in a single class (with the tolerance of one year accounting for misclassification errors in the source dataset) and change trajectories must change exactly once from the start class to the end class and otherwise remain stable. Anything else is considered erratic and not considered further. Any pixel which is classified as "developed" and any pixel which has no data in any given year is also not considered further.

Because outputs from these scripts remain fairly large, there are not saved in the GitHub repo.