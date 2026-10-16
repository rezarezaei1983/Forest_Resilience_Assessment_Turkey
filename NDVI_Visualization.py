"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script uses the NDVI rasters from 'calculate_Indicators.py' to        #
#    calculate the mean and variance of the NDVI files. The script also         #
#    visualizes the input rasters.                                              #
#                                                                               #
#>>> The mask value (-1000) and scaling factor (1000) of the input files will be# 
#    applied before conducting calculations.                                    #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2025
"""


import os
import glob
import pprint
import datetime
import numpy as np
import pandas as pd
import numpy.ma as ma
import rasterio as rio
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt


class NDVI:
    def __init__(self, input_rasters):
        self.input = input_rasters
        self.file = "unburned_pxl_removed_cropped_*.tif"
        self.ndvi_mask = 0
        
    def getStatistics(self, raster):
        with rio.open(raster, "r") as src:
            band_data = src.read(1)
            data_scale_factor = float(src.tags(bidx=0)["SCALE_FACTOR"])
                        
            masked_band = ma.masked_where(band_data < self.ndvi_mask, band_data)
            masked_band = masked_band * data_scale_factor
            mean = np.mean(masked_band).round(decimals=4)
            variance = np.var(masked_band).round(decimals=4)
                        
        return mean, variance
    
    def NDVImap(self, input_raster, study_area, pre_post, date):
        os.chdir(self.input)
        date = date.split("T")[0]
        datetime_date = datetime.datetime.strptime(date, '%Y%m%d').strftime("%b %d %Y")
        with rio.open(input_raster) as src:
            band_data = src.read(1)
            data_scale_factor = float(src.tags(bidx=0)["SCALE_FACTOR"])
    
            masked_band = ma.masked_where(band_data < self.ndvi_mask, band_data)
            masked_band = masked_band * data_scale_factor
            extent = src.bounds
            #raster_crs = src.crs
        
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
        extent_cartopy = [extent.left, extent.right, extent.bottom, extent.top]
        reversed_viridis = matplotlib.colormaps.get_cmap("viridis").reversed()

        im = ax.imshow(masked_band,
                       extent=extent_cartopy,
                       transform=ccrs.PlateCarree(),
                       cmap=reversed_viridis,
                       vmin=0,
                       vmax=1.0)
        
        height = ax.get_position().height
        #width = ax.get_position().width
        cax = fig.add_axes([ax.get_position().x0+0.005,
                            ax.get_position().y1-(height/5)-0.02,0.02, 
                            ax.get_position().height/5])
        cbar = plt.colorbar(im, ax=ax, cax=cax, orientation="vertical", 
                            ticks=np.arange(0.0, 1.01, 0.5), pad=0.05)

        cbar.ax.tick_params(labelsize=12)        
        plot_name = f"{study_area} {pre_post} NDVI\n({datetime_date})"
        ax.set_title(plot_name, fontsize=18)
        png_name = f"{study_area}_{pre_post}_NDVI_{date}.png"
        
        if os.path.exists(png_name):
            os.remove(png_name)
            
        output_path = os.path.join(self.input, png_name) 
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        #plt.show()

    def run(self):
        os.chdir(self.input)
        files = os.path.join(self.input, self.file) 
        infiles = glob.glob(files)
        
        for file in infiles:
            file_name = os.path.basename(file)[22:]
            study_area = file_name.split("_")[1]
            pre_post = file_name.split("_")[2]
            date = file_name.split("_")[-2]
            self.NDVImap(input_raster=file, study_area=study_area, 
                         pre_post=pre_post, date=date)
                     
        statistics = {}
        for file in infiles:
            file_name = os.path.basename(file)[22:]
            study_area = file_name.split("_")[1]
            if study_area not in statistics.keys():
                statistics[study_area] = {"Mean":{},
                                          "Variance":{}}
            mean, variance = self.getStatistics(file)
            date = file_name.split("_")[-2]
            statistics[study_area]["Mean"][date] = mean
            statistics[study_area]["Variance"][date] = variance

        for area, values in statistics.items():
            df = pd.DataFrame.from_dict(statistics[area], orient="index")
            df = df.transpose()
            csv_name = f"statisics_of_{area}.csv"
            if os.path.exists(csv_name):
                os.remove(csv_name)
              
            output_path = os.path.join(self.input, csv_name) 
            df.to_csv(output_path, index=True,header=True)
            
        print(f"\nThe outputs are written in:\n{self.input}")



#===================================== RUN ====================================
#input_rasters = "E:/Data/Indicators_revised/Indicators/AD/NDVI/"
#input_rasters = "E:/Data/Indicators_revised/Indicators/AN/NDVI/"
#input_rasters = "E:/Data/Indicators_revised/Indicators/ME/NDVI/"
input_rasters = "E:/Data/Indicators_revised/Indicators/MU/NDVI/"

ins = NDVI(input_rasters=input_rasters)
ins.run()