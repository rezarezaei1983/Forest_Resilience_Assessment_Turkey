"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script replaces the unburned areas in NDVI rasters by the mask value  #
#    (-1000). The outputs of this script will be used for all of the            #
#    calculations.                                                              #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2026
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import numpy.ma as ma
import rasterio as rio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from numpy import nan_to_num, subtract, add, divide, multiply

from _customCMAP import Linear, Segmented


class MaskUnburned:
    def __init__(self, ndvi_files, dnbr_file):
        self.ndvi = ndvi_files
        self.dnbr = dnbr_file
        self.mask_value = -1000
    
    def readfile(self, ndvi_file, burn_file):
        dir_path = os.path.dirname(ndvi_file)
        file_name = os.path.basename(ndvi_file)

        with rio.open(burn_file) as burn_src:
            burn_data = burn_src.read(1)
            burn_scale_factor = float(burn_src.tags(bidx=0)["SCALE_FACTOR"])
            burn_data = burn_data * burn_scale_factor

        with rio.open(ndvi_file) as ndvi_src:
            ndvi_data = ndvi_src.read(1)
            ndvi_arr = ndvi_data.copy()
            ndvi_arr[burn_data<=0.1] = self.mask_value       
            data_scale_factor = float(ndvi_src.tags(bidx=0)["SCALE_FACTOR"])
            out_meta = ndvi_src.meta
            src_transform = ndvi_src.transform
            out_meta.update({"driver": "GTiff",
                             "height": ndvi_arr.shape[0],
                             "width": ndvi_arr.shape[1],
                             "count": 1,
                             "transform": src_transform})
            file_name = f"unburned_pxl_removed_{file_name}"
            tiff_path = os.path.join(dir_path, f"{file_name}")

            with rio.open(tiff_path, "w", **out_meta) as dest:
                ndvi = ndvi_arr.copy()
                #ndvi = ndvi.filled(-1000)
                dest.write(ndvi, 1)
                dest.update_tags(SCALE_FACTOR=data_scale_factor)
            print("    ", file_name)
            
    def run(self):
        print("The outputs are saved as:")
        for ndvi_file in self.ndvi:
            self.readfile(ndvi_file, self.dnbr)
            
            
            

#===================================== RUN ====================================
### Adana area 3
ndvi_files = ["E:/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD03_Pre-fire_NDVI_20210724T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD03_Post-fire_NDVI_20210803T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD03_Post-fire_NDVI_20220729T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD03_Post-fire_NDVI_20230724T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD03_Post-fire_NDVI_20240728T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD03_Post-fire_NDVI_20250723T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD03_Post-fire_NDVI_20260728T081609_20m.tif"]
dnbr_file = "E:/Data/Indicators_revised/Indicators/AD/NBR/AD03_dNBR_20210803.tif"



ins = MaskUnburned(ndvi_files, dnbr_file)
ins.run()