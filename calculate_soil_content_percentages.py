"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script uses the outputs from the 'averaging_soil_layers.py' to        #
#    calculate the gridwise percentages of each soil content.                   #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2025
"""


import os
import numpy as np
import rasterio as rio

np.seterr(divide='ignore', invalid='ignore')


class SpoilContenPercenages:
    def __init__(self, input_content_rasters, output_dir, area_name):
        self.in_rasters = input_content_rasters
        self.outdir = output_dir
        self.area = area_name
        
    def Run(self):
        layers = []
        for area, values in self.in_rasters.items():
            with rio.open(values, "r") as src:
                band_data = src.read(1)
                layers.append(band_data)
                
        stacked_data = np.array(layers)
        aggregated_data = np.sum(stacked_data, axis=0)
        
        for area, values in self.in_rasters.items():
            with rio.open(values, "r") as src:
                band_data = src.read(1)
                src_transform = src.transform
                out_meta = src.meta
                scale_factor = float(src.tags(bidx=0)["SCALE_FACTOR"])
                data_name = src.tags(bidx=0)["DATA_NAME"]
                
                out_meta.update({"driver": "GTiff",
                                 "height": band_data.shape[0],
                                 "width": band_data.shape[1],
                                 "count": 1,
                                 "transform": src_transform})
                
                band = np.array(band_data)
                data = np.multiply(band, 100.0)
                data_in_perc = np.divide(data, aggregated_data)
                data_in_perc = data_in_perc.round(decimals=0)
                
                dt_name = data_name.lower().replace(" ", "-")
                out_tiff = os.path.join(self.outdir, f"{self.area}_{dt_name}-percentage.tif")     
                with rio.open(out_tiff, "w", **out_meta) as dest:
                    dest.write(data_in_perc, 1)
                    dest.update_tags(SCALE_FACTOR=scale_factor)
                    dest.update_tags(DATA_NAME=data_name)
                    dest.update_tags(UNIT="%")
                    
        print(f"\nThe outputs are written in:\n{self.outdir}")
                        
        
        

#===================================== RUN ====================================

### Adana
soil_content_rasters = {"Clay": "E:/Data/Soil_data/AD/Stacked_layers/AD_clay-content.tif",
                        "Sand": "E:/Data/Soil_data/AD/Stacked_layers/AD_sand-content.tif",
                        "Silt": "E:/Data/Soil_data/AD/Stacked_layers/AD_silt-content.tif"} 
output_dir = "E:/Data/Soil_data/AD/Stacked_layers"
area_name = "AD"     


ins = SpoilContenPercenages(input_content_rasters=soil_content_rasters, 
                            output_dir=output_dir, 
                            area_name=area_name)

ins.Run()