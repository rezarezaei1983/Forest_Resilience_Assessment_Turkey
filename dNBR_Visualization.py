"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script uses the NBR rasters from 'calculate_Indicators.py' to         #
#    calculate the dNBR values and save them in separatte GeoTIFF files. The    #
#    script also visualizes the dNBR maps.                                      #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2025
"""


import os
import datetime
import numpy as np
import pandas as pd
import numpy.ma as ma
import rasterio as rio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from numpy import nan_to_num, subtract, add, divide, multiply

from _customCMAP import Linear, Segmented


class dNBR:
    def __init__(self, pre_tiff, post_tiffs, cbar_position):
        self.ref = pre_tiff
        self.post = post_tiffs
        self.cbar_pos = cbar_position
        self.output_scale_factor = 1000
        self.mask_value = -1000
        
    def visualizeOnMap(self, extent, raster, scale_factor, title, plot_name, output_dir):
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
        extent_cartopy = [extent.left, extent.right, extent.bottom, extent.top]      
        raster = ma.masked_where(raster == self.mask_value, raster)
        raster = raster * scale_factor

        cmap, norm = Segmented().seg_dNBR()
        im = ax.imshow(raster,  
                       extent=extent_cartopy,
                       transform=ccrs.PlateCarree(),
                       cmap=cmap,
                       norm=norm) 
        
        if self.cbar_pos.casefold() == "v".casefold():
            cax = fig.add_axes([ax.get_position().x1+0.02,
                                ax.get_position().y0,0.02, 
                                ax.get_position().height])
            cbar = plt.colorbar(im, ax=ax, cax=cax, orientation="vertical", 
                                pad=0.05)
            cbar.ax.set_ylabel("Burn Severity", fontsize=14)                    
            cbar.ax.tick_params(labelsize=12)
            
        elif self.cbar_pos.casefold() == "h".casefold():
            cax = fig.add_axes([ax.get_position().x0,
                                ax.get_position().y0-0.06,
                                ax.get_position().width, 
                                0.02])
            cbar = fig.colorbar(im, ax=ax, cax=cax, orientation="horizontal", 
                                pad=0.05)
            cbar.ax.set_xlabel("Burn Severity", fontsize=15)
            cbar.ax.tick_params(labelsize=13)

        ax.set_title(title, fontsize=18)
        if os.path.exists(plot_name):
            os.remove(plot_name)
            
        output_path = os.path.join(output_dir, plot_name) 
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
                
    def calculate_dNBR(self, pre, post):
        dir_path = os.path.dirname(post)
        file_name = os.path.basename(post)
        area_name = file_name.split("_")[1]
        date = file_name.split("_")[-2].split("T")[0]
        datetime_date = datetime.datetime.strptime(date, '%Y%m%d').strftime("%b %d %Y")
        title = f"{area_name} dNBR\n({datetime_date})"
        plot_name = f"{area_name}_dNBR_{date}"

        stat = {"Mean": None, 
                "Median": None, 
                "Maximum": None,
                "Minimum": None}
        with rio.open(pre) as pre_src:
            pre_data = pre_src.read(1)
            masked_pre_data = ma.masked_where(pre_data == self.mask_value, pre_data)
            data_scale_factor = float(pre_src.tags(bidx=0)["SCALE_FACTOR"])
            masked_pre_data = masked_pre_data * data_scale_factor
            
        with rio.open(post) as post_src:
            post_data = post_src.read(1)
            extent = post_src.bounds
            out_meta = post_src.meta
            src_transform = post_src.transform
            
            masked_post_data = ma.masked_where(post_data == self.mask_value, post_data)
            data_scale_factor = float(post_src.tags(bidx=0)["SCALE_FACTOR"])
            masked_post_data = masked_post_data * data_scale_factor
            
            dNBR = subtract(masked_pre_data, masked_post_data)
            dNBR = dNBR.round(decimals=3) 
            
            out_meta.update({"driver": "GTiff",
                             "height": dNBR.shape[0],
                             "width": dNBR.shape[1],
                             "count": 1,
                             "transform": src_transform})
                        
            tiff_path = os.path.join(dir_path, f"{plot_name}.tif")
            with rio.open(tiff_path, "w", **out_meta) as dest:
                # remove cloud mask
                dNBR = dNBR.copy()
                dNBR = dNBR * self.output_scale_factor
                dNBR = dNBR.filled(-1000)
                dest.write(dNBR, 1)
                dest.update_tags(SCALE_FACTOR=1/self.output_scale_factor) 
            
            self.visualizeOnMap(extent=extent, raster=dNBR, 
                                scale_factor=1/self.output_scale_factor, title=title, 
                                plot_name=f"{plot_name}.png", output_dir=dir_path)
            
            masked_dNBR = ma.masked_where(dNBR == self.mask_value, dNBR)
            masked_dNBR = masked_dNBR * (1/self.output_scale_factor)
            stat["Mean"] = ma.mean(masked_dNBR).round(decimals=4)
            stat["Median"] = ma.median(masked_dNBR).round(decimals=4)
            stat["Maximum"] = ma.max(masked_dNBR).round(decimals=4) 
            stat["Minimum"] = ma.min(masked_dNBR).round(decimals=4) 
            
            df = pd.DataFrame.from_dict(stat, orient="index")
            df = df.transpose()
            csv_name = f"statisics_of_{area_name}_dNBR_{date}.csv"
            output_path = os.path.join(dir_path, csv_name)
            if os.path.exists(output_path):
                os.remove(output_path)
            
            df.to_csv(output_path, index=False, header=True)
            
    def run(self):       
        for post in self.post:
            self.calculate_dNBR(pre=self.ref, post=post)
            
        print(f"\nThe outputs are written in:\n{os.path.dirname(post)}")
        



#===================================== RUN ====================================

#################################### Adana ####################################
"""
### Adana area 1 
pre_tiff = "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD01_Pre-fire_NBR_20210724T081609_20m.tif"
post_tiffs = ["E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD01_Post-fire_NBR_20210803T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD01_Post-fire_NBR_20220729T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD01_Post-fire_NBR_20230724T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD01_Post-fire_NBR_20240728T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD01_Post-fire_NBR_20250723T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD01_Post-fire_NBR_20260728T081609_20m.tif"]
cbar_position="v"
"""
"""
### Adana area 2                                                          
pre_tiff = "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD02_Pre-fire_NBR_20210724T081609_20m.tif"
post_tiffs = ["E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD02_Post-fire_NBR_20210803T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD02_Post-fire_NBR_20220729T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD02_Post-fire_NBR_20230724T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD02_Post-fire_NBR_20240728T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD02_Post-fire_NBR_20250723T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD02_Post-fire_NBR_20260728T081609_20m.tif"]
cbar_position="h"
"""

### Adana area 3                                                          
pre_tiff = "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD03_Pre-fire_NBR_20210724T081609_20m.tif"                 
post_tiffs = ["E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD03_Post-fire_NBR_20210803T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD03_Post-fire_NBR_20220729T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD03_Post-fire_NBR_20230724T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD03_Post-fire_NBR_20240728T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD03_Post-fire_NBR_20250723T081609_20m.tif",
              "E:/Data/Indicators_revised/Indicators/AD/NBR/cropped_AD03_Post-fire_NBR_20260728T081609_20m.tif"]
cbar_position="v"
###############################################################################

ins = dNBR(pre_tiff=pre_tiff,
           post_tiffs=post_tiffs,
           cbar_position=cbar_position)
ins.run()