"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script crops soil data (outputs from the 'averagin_soil_layers.py'    #
#    and 'calculate_soil_content_percentae.py') using shapefiles.               #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2025
"""

import os
import fiona
import rasterio as rio
from rasterio.mask import mask


class CropSoilData:
    def __init__(self, soil_data, shapefile):
        self.soil_dt = soil_data
        self.shape = shapefile
    
    def cropRasterByShapefile(self, raster, area_name, shapefile):
        with rio.open(raster, "r") as src:
            with fiona.open(shapefile, "r") as shape:
                shapes = [feature['geometry'] for feature in shape]
            out_image, out_transform = mask(src, shapes, nodata=-1000, crop=True)
            out_meta = src.meta
            scale_factor = float(src.tags(bidx=0)["SCALE_FACTOR"])
            data_name = src.tags(bidx=0)["DATA_NAME"]
            unit = src.tags(bidx=0)["UNIT"]
            
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
            
            f = os.path.basename(raster)
            datatype = f.split("_")[-1]
            output_file = f"cropped_{area_name}_{datatype}"
            output_file_path = os.path.join(os.path.dirname(raster), output_file)
            with rio.open(output_file_path, "w", **out_meta) as dest:
                dest.write(out_image)
                dest.update_tags(SCALE_FACTOR=scale_factor)
                dest.update_tags(DATA_NAME=data_name)
                dest.update_tags(UNIT=unit)
                    
    def Run(self):
        for data_type in self.soil_dt:
            for area_name, shapefile in self.shape.items():
                self.cropRasterByShapefile(raster=data_type, 
                                           area_name=area_name, 
                                           shapefile=shapefile)
        
        print(f"\nThe outputs are written in:\n{os.path.dirname(data_type)}")
        



#===================================== RUN ====================================

#################################### Adana ####################################
soil_data = ["E:/Data/Soil_data/AD/Stacked_layers/AD_clay-content-percentage.tif",
             "E:/Data/Soil_data/AD/Stacked_layers/AD_sand-content-percentage.tif",
             "E:/Data/Soil_data/AD/Stacked_layers/AD_silt-content-percentage.tif",
             "E:/Data/Soil_data/AD/Stacked_layers/AD_nitrogen-content.tif",
             "E:/Data/Soil_data/AD/Stacked_layers/AD_organic-carbon-content.tif",
             "E:/Data/Soil_data/AD/Stacked_layers/AD_pH.tif",
             ]

shape = {"AD01": "E:/Data/Sentinel-2A/Shapefiles_revised/Adana_revised/temporal_adana.shp",
         "AD02": "E:/Data/Sentinel-2A/Shapefiles_revised/Adana_revised/temporal_adana_2.shp",
         "AD03": "E:/Data/Sentinel-2A/Shapefiles_revised/Adana_revised/temporal_adana_3.shp"}


ins = CropSoilData(soil_data=soil_data, 
                   shapefile=shape)

ins.Run()
