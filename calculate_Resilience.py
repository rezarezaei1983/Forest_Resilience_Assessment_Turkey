"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script uses NDVI outputs from 'calculate_Indicators.py' to calculate  #
#    the amplitude (only for the first post-fire input), malleability, recovery,#
#    and elasticity.                                                            #
#                                                                               #
#>>> The script outputs are: (1) a separate CSV file for each input file        #
#    containing the statistic summary of the calculated parameters, (2) a       #
#    separate GeoTIFF file for the calculated parameters, and (3) a separate    #
#    PNG image for each parameter.                                              #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2025
"""

import sys
import os
import datetime
import pandas as pd
import numpy as np
import numpy.ma as ma
import rasterio as rio
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
from numpy import nan_to_num, subtract, add, divide, multiply

from _customCMAP import Linear, Segmented

np.seterr(divide="ignore", invalid="ignore")


class Resilience:
    def __init__(self, reference_tiff, first_post_tiff, subsequent_post_tiffs,
                 max_amplitude, max_malleability, max_elasticity, max_recovery):
        self.ref = reference_tiff
        self.first_post = first_post_tiff
        self.sub_post = subsequent_post_tiffs
        self.max_amp = max_amplitude
        self.max_mal = max_malleability
        self.max_elst = max_elasticity
        self.max_rcvry = max_recovery
        self.ndvi_mask = 0
        self.output_scale_factor = 1000
        
    def visualizeOnMap(self, extent, raster, title, plot_name, output_dir, 
                       upper_limit, idx_type):  # elasticity
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, dpi=300)
        extent_cartopy = [extent.left, extent.right, extent.bottom, extent.top]
        
        if idx_type == "elasticity":            
            im = ax.imshow(raster,
                           extent=extent_cartopy,
                           transform=ccrs.PlateCarree(),
                           cmap=plt.get_cmap(Linear().BuGnMagma1().reversed()), 
                           vmin=0, 
                           vmax=upper_limit)
            ticks=[0, 70, upper_limit]
        elif idx_type == "recovery":
            im = ax.imshow(raster,
                           extent=extent_cartopy,
                           transform=ccrs.PlateCarree(),
                           cmap=plt.get_cmap(Linear().BuGnMagma2().reversed()), 
                           vmin=-1, 
                           vmax=upper_limit)
            ticks=[-1, 0, upper_limit]
        elif idx_type == "recovery percent":
            im = ax.imshow(raster,
                           extent=extent_cartopy,
                           transform=ccrs.PlateCarree(),
                           cmap=plt.get_cmap(Linear().BuGnMagma2().reversed()), 
                           vmin=np.nanmin(raster), 
                           vmax=np.nanmax(raster))
            ticks=[np.nanmin(raster), np.nanmax(raster)]
        else:
            im = ax.imshow(raster,
                           extent=extent_cartopy,
                           transform=ccrs.PlateCarree(),
                           cmap=matplotlib.colormaps.get_cmap("magma").reversed(),
                           vmin=0,  # vmin=np.nanmin(raster)
                           vmax=upper_limit)  # vmax=np.nanmax(raster)
            ticks=[-1, 0, upper_limit]
        
        height = ax.get_position().height
        #width = ax.get_position().width
        cax = fig.add_axes([ax.get_position().x0+0.005,
                            ax.get_position().y1-(height/5)-0.02,0.02, 
                            ax.get_position().height/5])
        cbar = plt.colorbar(im, ax=ax, cax=cax, orientation="vertical", 
                            ticks=ticks, pad=0.05)

        cbar.ax.tick_params(labelsize=12)        
        ax.set_title(title, fontsize=18)
        if os.path.exists(plot_name):
            os.remove(plot_name)
            
        output_path = os.path.join(output_dir, plot_name) 
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
            
    def amplitude(self, pre, first_post):
        dir_path = os.path.dirname(first_post)
        file_name = os.path.basename(first_post)[22:]
        area_name = file_name.split("_")[1]
        date = file_name.split("_")[-2].split("T")[0]
        datetime_date = datetime.datetime.strptime(date, '%Y%m%d').strftime("%b %d %Y")
        title = f"{area_name} Amplitude\n({datetime_date})"
        plot_name = f"{area_name}_amplitude_{date}"
        
        stat = {"Mean": None, 
                "Median": None, 
                "Maximum": None,
                "Minimum": None}
        with rio.open(pre) as pre_src:
            pre_data = pre_src.read(1)
            data_scale_factor = float(pre_src.tags(bidx=0)["SCALE_FACTOR"])
            
            masked_pre_data = ma.masked_where(pre_data < self.ndvi_mask, pre_data)
            masked_pre_data = masked_pre_data * data_scale_factor
            
        with rio.open(first_post) as post_src:
            post_data = post_src.read(1)
            extent = post_src.bounds
            out_meta = post_src.meta
            src_transform = post_src.transform
            data_scale_factor = float(post_src.tags(bidx=0)["SCALE_FACTOR"])

            masked_post_data = ma.masked_where(post_data < self.ndvi_mask, post_data)
            masked_post_data = masked_post_data * data_scale_factor
            amplitude = abs(subtract(masked_post_data, masked_pre_data))
            
            out_meta.update({"driver": "GTiff",
                             "height": amplitude.shape[0],
                             "width": amplitude.shape[1],
                             "count": 1,
                             "transform": src_transform})
            
            tiff_path = os.path.join(dir_path, f"{plot_name}.tif")               
            with rio.open(tiff_path, "w", **out_meta) as dest:
                # remove cloud mask
                amp = amplitude.copy()
                amp = amp * self.output_scale_factor
                amp = amp.filled(-1000)
                dest.write(amp, 1)
                dest.update_tags(SCALE_FACTOR=1/self.output_scale_factor)                
            
            self.visualizeOnMap(extent=extent, raster=amplitude, title=title, 
                                plot_name=f"{plot_name}.png", output_dir=dir_path, 
                                upper_limit=self.max_amp, idx_type="amplitude")
            
            stat["Mean"] = ma.mean(amplitude).round(decimals=4) 
            stat["Median"] = ma.median(amplitude).round(decimals=4)
            stat["Maximum"] = ma.max(amplitude).round(decimals=4) 
            stat["Minimum"] = ma.min(amplitude).round(decimals=4) 
            
            df = pd.DataFrame.from_dict(stat, orient="index")
            df = df.transpose()
            csv_name = f"statisics_of_{area_name}_Amplitude_{date}.csv"
            output_path = os.path.join(dir_path, csv_name)
            if os.path.exists(output_path):
                os.remove(output_path)
                
            df.to_csv(output_path, index=False, header=True)
    
    def malleability(self, pre, first_post, sub_post):
        dir_path = os.path.dirname(sub_post)
        file_name = os.path.basename(sub_post)[22:]
        area_name = file_name.split("_")[1]
        date = file_name.split("_")[-2].split("T")[0]
        datetime_date = datetime.datetime.strptime(date, '%Y%m%d').strftime("%b %d %Y")
        title = f"{area_name} Malleability\n({datetime_date})"
        plot_name = f"{area_name}_malleability_{date}"
        
        stat = {"Mean": None, 
                "Median": None, 
                "Maximum": None,
                "Minimum": None}
        with rio.open(pre) as pre_src:
            pre_data = pre_src.read(1)
            data_scale_factor = float(pre_src.tags(bidx=0)["SCALE_FACTOR"])
            
            masked_pre_data = ma.masked_where(pre_data < self.ndvi_mask, pre_data)
            masked_pre_data = masked_pre_data * data_scale_factor

        with rio.open(first_post) as post_src:
            post_data = post_src.read(1)
            data_scale_factor = float(post_src.tags(bidx=0)["SCALE_FACTOR"])
            
            masked_post_data = ma.masked_where(post_data < self.ndvi_mask, post_data)
            masked_post_data = masked_post_data * data_scale_factor
            amplitude = abs(subtract(masked_post_data, masked_pre_data))
                    
        with rio.open(sub_post) as sub_src:
            sub_post = sub_src.read(1)
            extent = sub_src.bounds
            out_meta = sub_src.meta
            src_transform = sub_src.transform
            data_scale_factor = float(sub_src.tags(bidx=0)["SCALE_FACTOR"])
            
            masked_sub_data = ma.masked_where(sub_post < self.ndvi_mask, sub_post)
            masked_sub_data = masked_sub_data * data_scale_factor
            
            x = subtract(masked_sub_data, masked_pre_data)
            malleability = subtract(amplitude, x)

            out_meta.update({"driver": "GTiff",
                             "height": malleability.shape[0],
                             "width": malleability.shape[1],
                             "count": 1,
                             "transform": src_transform})
            
            tiff_path = os.path.join(dir_path, f"{plot_name}.tif")
            with rio.open(tiff_path, "w", **out_meta) as dest:
                # remove cloud mask
                mall = malleability.copy()
                mall = mall * self.output_scale_factor
                mall = mall.filled(-1000)
                dest.write(mall, 1)
                dest.update_tags(SCALE_FACTOR=1/self.output_scale_factor)            
                
            self.visualizeOnMap(extent=extent, raster=malleability, title=title, 
                                plot_name=f"{plot_name}.png", output_dir=dir_path, 
                                upper_limit=self.max_mal, idx_type="malleability")
            
            stat["Mean"] = ma.mean(malleability).round(decimals=4) 
            stat["Median"] = ma.median(malleability).round(decimals=4)
            stat["Maximum"] = ma.max(malleability).round(decimals=4) 
            stat["Minimum"] = ma.min(malleability).round(decimals=4) 
            
            df = pd.DataFrame.from_dict(stat, orient="index")
            df = df.transpose()
            csv_name = f"statisics_of_{area_name}_Malleability_{date}.csv"
            output_path = os.path.join(dir_path, csv_name)
            if os.path.exists(output_path):
                os.remove(output_path)
            
            df.to_csv(output_path, index=False, header=True)
            
    def elasticity(self, pre, sub_post):
        dir_path = os.path.dirname(sub_post)
        file_name = os.path.basename(sub_post)[22:]
        area_name = file_name.split("_")[1]
        date = file_name.split("_")[-2].split("T")[0]
        datetime_date = datetime.datetime.strptime(date, '%Y%m%d').strftime("%b %d %Y")
        title = f"{area_name} Elasticity\n({datetime_date})"
        plot_name = f"{area_name}_elasticity_{date}"
        
        stat = {"Mean": None, 
                "Median": None, 
                "Maximum": None,
                "Minimum": None}
        with rio.open(pre) as pre_src:
            pre_data = pre_src.read(1)
            data_scale_factor = float(pre_src.tags(bidx=0)["SCALE_FACTOR"])
            
            masked_pre_data = ma.masked_where(pre_data < self.ndvi_mask, pre_data)
            masked_pre_data = masked_pre_data * data_scale_factor
        
        with rio.open(sub_post) as sub_src:
            sub_post = sub_src.read(1)
            extent = sub_src.bounds
            out_meta = sub_src.meta
            src_transform = sub_src.transform
            data_scale_factor = float(sub_src.tags(bidx=0)["SCALE_FACTOR"])
            
            masked_sub_data = ma.masked_where(sub_post < self.ndvi_mask, sub_post)
            masked_sub_data = masked_sub_data * data_scale_factor
            
            x = multiply(masked_sub_data, 100)
            elasticity = divide(x, masked_pre_data)
            
            out_meta.update({"driver": "GTiff",
                             "height": elasticity.shape[0],
                             "width": elasticity.shape[1],
                             "count": 1,
                             "transform": src_transform})
                        
            tiff_path = os.path.join(dir_path, f"{plot_name}.tif")
            with rio.open(tiff_path, "w", **out_meta) as dest:
                # remove cloud mask
                elst = elasticity.copy()
                elst = elst * self.output_scale_factor
                elst = elst.filled(-1000)
                dest.write(elst, 1)
                dest.update_tags(SCALE_FACTOR=1/self.output_scale_factor) 
            
            self.visualizeOnMap(extent=extent, raster=elasticity, title=title, 
                                plot_name=f"{plot_name}.png", output_dir=dir_path, 
                                upper_limit=self.max_elst, idx_type="elasticity")
            
            stat["Mean"] = ma.mean(elasticity).round(decimals=4) 
            stat["Median"] = ma.median(elasticity).round(decimals=4)
            stat["Maximum"] = ma.max(elasticity).round(decimals=4) 
            stat["Minimum"] = ma.min(elasticity).round(decimals=4) 
            
            df = pd.DataFrame.from_dict(stat, orient="index")
            df = df.transpose()
            csv_name = f"statisics_of_{area_name}_Elasticity_{date}.csv"
            output_path = os.path.join(dir_path, csv_name)
            if os.path.exists(output_path):
                os.remove(output_path)
            
            df.to_csv(output_path, index=False, header=True)
    
    def recoveryPercentage(self, pre, first_post, sub_post):
        dir_path = os.path.dirname(sub_post)
        file_name = os.path.basename(sub_post)[22:]
        area_name = file_name.split("_")[1]
        date = file_name.split("_")[-2].split("T")[0]
        datetime_date = datetime.datetime.strptime(date, '%Y%m%d').strftime("%b %d %Y")
        title = f"{area_name} Recovery (%)\n({datetime_date})"
        plot_name = f"{area_name}_recovery-percentage_{date}"

        stat = {"Mean": None, 
                "Median": None, 
                "Maximum": None,
                "Minimum": None}
        
        with rio.open(pre) as pre:
            pre_data = pre.read(1)
            data_scale_factor = float(pre.tags(bidx=0)["SCALE_FACTOR"])
            masked_pre_data = ma.masked_where(pre_data < self.ndvi_mask, pre_data)
            masked_pre_data = masked_pre_data * data_scale_factor
            
        with rio.open(first_post) as post_src:
            post_data = post_src.read(1)
            data_scale_factor = float(post_src.tags(bidx=0)["SCALE_FACTOR"])
            masked_post_data = ma.masked_where(post_data < self.ndvi_mask, post_data)
            masked_post_data = masked_post_data * data_scale_factor
            
        with rio.open(sub_post) as sub_src:
            sub_post = sub_src.read(1)
            extent = sub_src.bounds
            out_meta = sub_src.meta
            src_transform = sub_src.transform
            data_scale_factor = float(sub_src.tags(bidx=0)["SCALE_FACTOR"])
            masked_sub_data = ma.masked_where(sub_post < self.ndvi_mask, sub_post)
            masked_sub_data = masked_sub_data * data_scale_factor
            
            decrease = subtract(masked_pre_data, masked_post_data)            
            recovery = subtract(masked_sub_data, masked_post_data)
            recovery = recovery * 100            
            recovery_percent = divide(recovery, decrease)
            
            out_meta.update({"driver": "GTiff",
                             "height": masked_sub_data.shape[0],
                             "width": masked_sub_data.shape[1],
                             "count": 1,
                             "transform": src_transform})
            
            tiff_path = os.path.join(dir_path, f"{plot_name}.tif")
            if os.path.exists(tiff_path):
                os.remove(tiff_path)
            
            with rio.open(tiff_path, "w", **out_meta) as dest:
                # remove cloud mask
                rcvry = recovery_percent.copy()
                rcvry = rcvry * 1
                rcvry = rcvry.filled(-1000)
                dest.write(rcvry, 1)
                dest.update_tags(SCALE_FACTOR=1)            
            
            if os.path.exists(os.path.join(dir_path, f"{plot_name}.png")):
                os.remove(os.path.join(dir_path, f"{plot_name}.png"))

            self.visualizeOnMap(extent=extent, raster=recovery_percent, title=title, 
                                plot_name=f"{plot_name}.png", output_dir=dir_path, 
                                upper_limit=self.max_rcvry, idx_type="recovery percent")# <<<
            
            stat["Mean"] = ma.mean(recovery_percent).round(decimals=4) 
            stat["Median"] = ma.median(recovery_percent).round(decimals=4)
            stat["Maximum"] = ma.max(recovery_percent).round(decimals=4) 
            stat["Minimum"] = ma.min(recovery_percent).round(decimals=4) 
            
            df = pd.DataFrame.from_dict(stat, orient="index")
            df = df.transpose()
            csv_name = f"statisics_of_{area_name}_Recovery-Percentage_{date}.csv"
            output_path = os.path.join(dir_path, csv_name)
            if os.path.exists(output_path):
                os.remove(output_path)
            
            df.to_csv(output_path, index=False, header=True)
    
    def recovery(self, first_post, sub_post):
        dir_path = os.path.dirname(sub_post)
        file_name = os.path.basename(sub_post)[22:]
        area_name = file_name.split("_")[1]
        date = file_name.split("_")[-2].split("T")[0]
        datetime_date = datetime.datetime.strptime(date, '%Y%m%d').strftime("%b %d %Y")
        title = f"{area_name} Recovery\n({datetime_date})"
        plot_name = f"{area_name}_recovery_{date}"

        stat = {"Mean": None, 
                "Median": None, 
                "Maximum": None,
                "Minimum": None}
        with rio.open(first_post) as post_src:
            post_data = post_src.read(1)
            data_scale_factor = float(post_src.tags(bidx=0)["SCALE_FACTOR"])
            masked_post_data = ma.masked_where(post_data < self.ndvi_mask, post_data)
            masked_post_data = masked_post_data * data_scale_factor

        with rio.open(sub_post) as sub_src:
            sub_post = sub_src.read(1)
            extent = sub_src.bounds
            out_meta = sub_src.meta
            src_transform = sub_src.transform
            data_scale_factor = float(sub_src.tags(bidx=0)["SCALE_FACTOR"])
            masked_sub_data = ma.masked_where(sub_post < self.ndvi_mask, sub_post)
            masked_sub_data = masked_sub_data * data_scale_factor
            
            recovery = subtract(masked_sub_data, masked_post_data)

            out_meta.update({"driver": "GTiff",
                             "height": masked_sub_data.shape[0],
                             "width": masked_sub_data.shape[1],
                             "count": 1,
                             "transform": src_transform})
            
            tiff_path = os.path.join(dir_path, f"{plot_name}.tif")
            with rio.open(tiff_path, "w", **out_meta) as dest:
                # remove cloud mask
                rcvry = recovery.copy()
                rcvry = rcvry * self.output_scale_factor
                rcvry = rcvry.filled(-1000)
                dest.write(rcvry, 1)
                dest.update_tags(SCALE_FACTOR=1/self.output_scale_factor)            

            self.visualizeOnMap(extent=extent, raster=recovery, title=title, 
                                plot_name=f"{plot_name}.png", output_dir=dir_path, 
                                upper_limit=self.max_rcvry, idx_type="recovery")
            
            stat["Mean"] = ma.mean(recovery).round(decimals=4) 
            stat["Median"] = ma.median(recovery).round(decimals=4)
            stat["Maximum"] = ma.max(recovery).round(decimals=4) 
            stat["Minimum"] = ma.min(recovery).round(decimals=4) 
            
            df = pd.DataFrame.from_dict(stat, orient="index")
            df = df.transpose()
            csv_name = f"statisics_of_{area_name}_Recovery_{date}.csv"
            output_path = os.path.join(dir_path, csv_name)
            if os.path.exists(output_path):
                os.remove(output_path)
            
            df.to_csv(output_path, index=False, header=True)
       
    def run(self):
        self.amplitude(pre=self.ref, first_post=self.first_post)
        
        for sub in self.sub_post:
            self.malleability(pre=self.ref, first_post=self.first_post, sub_post=sub)
            self.elasticity(pre=self.ref, sub_post=sub)
            #self.recovery(first_post=self.first_post, sub_post=sub)
            self.recoveryPercentage(pre=self.ref, first_post=self.first_post, sub_post=sub)
            
        print(f"\nThe outputs are written in:\n{os.path.dirname(sub)}")
        



#===================================== RUN ====================================

#################################### Adana ####################################
"""
### Adana area 1                                                                    
reference_tiff = "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD01_Pre-fire_NDVI_20210724T081609_20m.tif"
first_post_tiff = "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD01_Post-fire_NDVI_20210803T081609_20m.tif"
sub_post_tiffs = ["E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD01_Post-fire_NDVI_20220729T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD01_Post-fire_NDVI_20230724T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD01_Post-fire_NDVI_20240728T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD01_Post-fire_NDVI_20250723T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD01_Post-fire_NDVI_20260728T081609_20m.tif"]
"""
"""
### Adana area 2
reference_tiff = "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD02_Pre-fire_NDVI_20210724T081609_20m.tif"
first_post_tiff = "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD02_Post-fire_NDVI_20210803T081609_20m.tif"
sub_post_tiffs = ["E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD02_Post-fire_NDVI_20220729T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD02_Post-fire_NDVI_20230724T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD02_Post-fire_NDVI_20240728T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD02_Post-fire_NDVI_20250723T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD02_Post-fire_NDVI_20260728T081609_20m.tif"]
"""

### Adana area 3
reference_tiff = "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD03_Pre-fire_NDVI_20210724T081609_20m.tif"
first_post_tiff = "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD03_Post-fire_NDVI_20210803T081609_20m.tif"
sub_post_tiffs = ["E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD03_Post-fire_NDVI_20220729T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD03_Post-fire_NDVI_20230724T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD03_Post-fire_NDVI_20240728T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD03_Post-fire_NDVI_20250723T081609_20m.tif",
                  "E:/Data/Indicators_revised/Indicators/AD/NDVI/unburned_pxl_removed_cropped_AD03_Post-fire_NDVI_20260728T081609_20m.tif"]

###############################################################################

ins = Resilience(reference_tiff=reference_tiff,
                 first_post_tiff=first_post_tiff,
                 subsequent_post_tiffs=sub_post_tiffs,
                 max_amplitude=1.0,
                 max_malleability=1.2,
                 max_elasticity=100,
                 max_recovery=1.0)

ins.run()