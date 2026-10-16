"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script upscales the malleability, nitrogen content, organic carbon    #
#    content, and pH rasters to the resolution of 250m using the grid size of   #
#      soil data ('destination_rasters').                                       #
#                                                                               #
#>>> Both of the malleability and destination rasters must be in EPSG:4326-WSG84#
#    projection.                                                                #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2025
"""

import os
import rasterio as rio
from rasterio.enums import Resampling


class Resample:
    def __init__(self, rasters_to_resample, destination_rasters, output_dir):
        self.input_rasters = rasters_to_resample
        self.destination = destination_rasters
        self.outdir = output_dir
        
    def RasterResampling(self, input_raster, area_name):
        input_rst = os.path.basename(input_raster)
        output_rst = f"resampled_{input_rst}"
        output_dir = os.path.join(self.outdir, output_rst)

        with rio.open(self.destination[area_name], "r") as dest_src:
            target_resolution_x, target_resolution_y = dest_src.res
            target_transform = dest_src.transform
            target_width, target_height = dest_src.width, dest_src.height
        
        with rio.open(input_raster) as src:
            data = src.read(out_shape=(src.count, target_height, target_width),
                            resampling=Resampling.average)
            scale_factor = float(src.tags(bidx=0)["SCALE_FACTOR"])
            
        new_meta = src.meta.copy()
        new_meta.update({"driver": "GTiff",
                         "height": target_height,
                         "width": target_width,
                         "transform": target_transform,
                         "res": (target_resolution_x, target_resolution_y)})
            
        with rio.open(output_dir, "w", **new_meta) as dst:
            dst.write(data)
            dst.update_tags(SCALE_FACTOR=scale_factor)
        
    def Run(self):
        for variable, areas in self.input_rasters.items():
            for area_name, raster_path in self.input_rasters[variable].items():
                self.RasterResampling(input_raster=raster_path, 
                                          area_name=area_name)
            
        print(f"Outputs saved to:\n{self.outdir}")
        




#===================================== RUN ====================================

#################################### Adana ####################################
input_rasters = {   
    "Malleability": {"AD01": "E:/Data/Indicators_revised/Indicators/AD/NDVI/AD01_malleability_20250723.tif",
                     "AD02": "E:/Data/Indicators_revised/Indicators/AD/NDVI/AD02_malleability_20250723.tif",
                     "AD03": "E:/Data/Indicators_revised/Indicators/AD/NDVI/AD03_malleability_20250723.tif"},
    "Nitrogen Content": {"AD01": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD01_nitrogen-content.tif",
                         "AD02": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD02_nitrogen-content.tif",
                         "AD03": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD03_nitrogen-content.tif"},
    "Organic Carbon Content": {"AD01": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD01_organic-carbon-content.tif",
                               "AD02": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD02_organic-carbon-content.tif",
                               "AD03": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD03_organic-carbon-content.tif"},
    "pH": {"AD01": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD01_pH.tif",
           "AD02": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD02_pH.tif",
           "AD03": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD03_pH.tif"}
                }

destination_raster = {"AD01": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD01_clay-content-percentage.tif",
                      "AD02": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD02_clay-content-percentage.tif",
                      "AD03": "E:/Data/Soil_data/AD/Stacked_layers/cropped_AD03_clay-content-percentage.tif"}

output_dir = "E:/Data/Soil_data/AD/Stacked_layers/"

###############################################################################




ins = Resample(rasters_to_resample=input_rasters, 
               destination_rasters=destination_raster,
               output_dir=output_dir)

ins.Run()