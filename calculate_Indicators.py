"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script calculates the NDVI and NBR indices using the Band04, Band08,  #
#    and Band12 of the Sentinel-2A imagery.                                     #
#                                                                               #
#>>> The script uses the ZIP files. So, the Sentinel-2A images mustn't unzip.   #
#    The pre-fire file and post-fire TIFF files must be placed in different     #
#    directories. Also, the shapefiles are provided by the QGIS.                #
#                                                                               #
#>>> The script is unable to create the mosaic of the input files.              #
#                                                                               #
#>>> The B04 band is in 10m and 20m. The B08 band is in 10m and the B12 band is # 
#    only given in 20m. The script resamples B08 from 10m resolution to 20m and # 
#    saves the raster as a separate GeoTIFF file. The bands with 20m resolution #
#    are used to calculate the indicators.                                      #
#                                                                               #
#>>> For Sentinel-2, the relation between precision the digital numbers (DN) and#
#    REFLECTANCE is: DN = 10000 * REFLECTANCE. So, the band values are divided  # 
#    by the scaling factor (self.sentinel2_scaling_facto).                      #
#                                                                               #
#>>> The script replaces grids containing cloud cover, shadows, and missing data#             
#    with (-1000) using the classes in the 'SCL_20m' band. So, before using the #
#    output files, the raster must be masked by -1000.                          #
#                                                                               #
#>>> The resolution of the NDVI and NBR rasters in the output files is 20m.     #
#                                                                               #
#>>> The scale factor of the data in the output GeoTIFF files is equal to 1000  #
#    ('self.output_scale_factor'), which is written in TAG of the output files. #
#    So, before using the data in the output files the values must be divided   #
#    by 1000.                                                                   #
#                                                                               #   
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2025
"""


import os
import fiona
import shutil
import warnings
import numpy as np
import numpy.ma as ma
from numpy import nan_to_num, subtract, add, divide, multiply
from pathlib import Path
from zipfile import ZipFile
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.mask import mask
from rasterio.plot import show_hist
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
np.seterr(divide='ignore', invalid='ignore')


class Indicator:
    def __init__(self, pre_fire_zip, post_fire_zips_dir, 
                 shapefile_dir, output_dir, region_name):
        self.pre = pre_fire_zip
        self.post = post_fire_zips_dir
        self.shpdir = shapefile_dir
        self.outdir = output_dir
        self.region = region_name
        self.zip = "*.SAFE.zip"
        self.image_extension = ".jp2"
        self.base_dir = "Indicators"
        self.nodata_value = -9999
        self.sentinel2_scaling_factor = 10000
        self.output_scale_factor = 1000
    
    def resample10mto20m(self, res_10m, pre_post):
        output_10m_path = os.path.basename(res_10m)
        output_20m_path = output_10m_path.replace(output_10m_path[-7:], "20m.tif")
        path = os.path.dirname(res_10m)
        base_dir = path.split(".zip")[0] + ".zip" 
        base_dir = Path(base_dir)  
        base_dir =os.fspath(Path(base_dir.parent.parent).resolve())
        output_dir = os.path.join(base_dir, output_20m_path)
        
        with rio.open(f"zip://{res_10m}") as src:
            profile = src.profile
            profile.update(width=src.width // 2,
                           height=src.height // 2,
                           transform=src.transform * src.transform.scale(2, 2))
            # Resample to 20m resolution
            with rio.open(output_dir, "w", **profile) as dst:
                reproject(source=rio.band(src, 1),
                          destination=rio.band(dst, 1),
                          resampling=Resampling.average)
        
        zip_name = "resampled_" + output_20m_path[:-4] + ".zip"
        os.chdir(base_dir)
        ZipFile(zip_name, mode="w").write(output_20m_path)
        os.remove(output_20m_path)
        output_dir = os.path.join(base_dir, zip_name, output_20m_path)
        
        if pre_post == "Pre-fire":
            print("The B08 band is resampled from 10m to 20m resolution and the "\
                  "output raster is saved in the below file:\n", output_dir, "\n")
 
        return output_dir
    
    def getRasterData(self, band):
        with rio.open(f"zip://{band}") as scl_src:
            band_data = scl_src.read(1)         
           
        return band_data
        
    def maskCloudCover(self, scl_band, band_data):
        copied_band_data = band_data.copy()
        with rio.open(f"zip://{scl_band}") as scl_src:   
            scl_data = scl_src.read(1)    
            # Mask clouds (8, 9, 10), shadows (3), and missing data (0) based on SCL classes
            cloud_shadow_mask = np.isin(scl_data, [0, 3, 8, 9, 10])            
            copied_band_data[cloud_shadow_mask] = self.nodata_value                        
            masked_band = np.ma.masked_where(copied_band_data < self.nodata_value + 1000, band_data)

            return masked_band  
    
    def createPNG(self, date, pre_post, ind_type, base_dir, destination_array):
        d = date.split("T")[0]
        im_name = f"{pre_post}_{ind_type}_{date}_20m.png"
        path = os.path.join(base_dir, ind_type, im_name)
        mean = np.mean(destination_array).round(decimals=2)
        maxim = np.max(destination_array).round(decimals=2)
        minim = np.min(destination_array).round(decimals=2)   
        
        plt.imshow(destination_array, cmap='RdYlGn')
        plt.colorbar(label=f"{ind_type} (\u00D7{1/self.output_scale_factor})")
        plt.title(f"{pre_post} {ind_type} ({d})\nMean={mean}, Max={maxim}, Min={minim}")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        
        #show_hist(destination_array, bins=50, lw=0.0, stacked=False, alpha=0.3,
        #          histtype='stepfilled', title="Histogram")  
        
    def cropRasterByShapefile(self, raster, lower_limit):
        area_list = []
        with rio.open(raster, "r") as src:
            for area_name, shapefile in self.shpdir.items():
                with fiona.open(shapefile, "r") as shape:
                    shapes = [feature['geometry'] for feature in shape]
                out_image, out_transform = mask(src, shapes, nodata=-1000, crop=True)
                out_meta = src.meta
                
                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
                
                f = os.path.basename(raster)
                area = area_name.split(" ")[-1]
                area_list.append(area)
                output_file = f"cropped_{self.region}{area}_{f}"
                output_file_path = os.path.join(os.path.dirname(raster), output_file)
                
                out_image = np.where(out_image < lower_limit, -1000, out_image)
                with rio.open(output_file_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                    dest.update_tags(SCALE_FACTOR=1/self.output_scale_factor)
        
        separator = " " 
        areas = separator.join(area_list).replace(" ", "-")
        out_path = os.path.dirname(raster)
        output_name = str(os.path.basename(raster))[:-4] 
        old_out_name = f"{output_name}.png"
        new_out_name = f"Area-{areas}_{output_name}.png"     
        old_out_path = os.path.join(out_path, old_out_name)
        new_out_path = os.path.join(out_path, new_out_name)
        if os.path.exists(new_out_path):
            os.remove(new_out_path)
        os.rename(old_out_path, new_out_path) 

        for fname in os.listdir(out_path):
            if fname.startswith("Pre-fire_") or fname.startswith("Post-fire_"):
                os.remove(os.path.join(out_path, fname))
                
    def createReprojectedGeotiff(self, source_crs, ind_array, ind_type, pre_post):
        date = os.path.basename(source_crs).split("_")[1]
        with rio.open(f"zip://{source_crs}") as src:
            target_crs = "EPSG:4326"
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)

            kwargs = src.meta.copy()
            kwargs.update({"crs": target_crs,
                           "transform": transform,
                           "width": width,
                           "height": height,
                           "dtype": ind_array.dtype})
                     
            base_dir = os.path.join(self.outdir, self.base_dir, self.region)
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
                
            os.chdir(base_dir)
            if not os.path.exists(ind_type):
                os.mkdir(ind_type)
            
            file_name = f"{pre_post}_{ind_type}_{date}_20m.tif"
            output_path = os.path.join(base_dir, ind_type, file_name)
            if os.path.exists(output_path):
                os.remove(output_path)
                
            with rio.open(output_path, "w", **kwargs) as dst:
                lower_limit = {"NDVI":0,
                               "NBR":-500}
                # remove cloud mask
                ind_array = ind_array.filled(-1000)
                destination_array = np.empty((height, width), dtype=ind_array.dtype)
                reproject(source=ind_array,
                          destination=destination_array,
                          src_transform=src.transform,
                          src_crs=src.crs,
                          dst_transform=transform,
                          dst_crs=target_crs,
                          dst_nodata=self.nodata_value,
                          resampling=Resampling.nearest)
                destination_array = np.nan_to_num(destination_array, nan=self.nodata_value)
                destination_array = np.where(destination_array < lower_limit[ind_type], 
                                             -1000, destination_array)
                dst.write(destination_array, 1)
                dst.update_tags(SCALE_FACTOR=1/self.output_scale_factor)   
                
                self.createPNG(date, pre_post, ind_type, base_dir, destination_array)
        
        self.cropRasterByShapefile(raster=output_path, lower_limit=lower_limit[ind_type])
                                            
    def calculateIndicators(self, zipped_image, pre_post):
        cloud_20m = []
        band4_20m = []
        band8_20m = []
        band8_10m = []
        band12_20m = []
        with ZipFile(zipped_image, "r") as archive:
            list_of_files = archive.namelist()
            for elem in list_of_files:
                if os.path.splitext(elem)[-1] == self.image_extension:
                    if elem[-11:-4] == "B04_20m":
                        b04 = os.path.join(zipped_image, elem)
                        band4_20m.append(b04)                 
                    elif elem[-11:-4] == "B08_20m":
                        b08 = os.path.join(zipped_image, elem)
                        band8_20m.append(b08)
                    elif elem[-11:-4] == "B08_10m":
                        b08 = os.path.join(zipped_image, elem)
                        band8_10m.append(b08)
                    elif elem[-11:-4] == "B12_20m":
                        b12 = os.path.join(zipped_image, elem)
                        band12_20m.append(b12)
                    elif elem[-11:-4] == "SCL_20m":
                        scl = os.path.join(zipped_image, elem)
                        cloud_20m.append(scl)
                    else:
                        pass

        if len(band8_20m) == 0:
            b8_20m = self.resample10mto20m(res_10m=band8_10m[0], pre_post=pre_post)
            band8_20m.append(b8_20m)
        else:
            pass
        
        r = self.getRasterData(band=band4_20m[0])
        r = r / self.sentinel2_scaling_factor
        r = self.maskCloudCover(scl_band=cloud_20m[0], band_data=r)
        nir = self.getRasterData(band=band8_20m[0])
        nir = nir / self.sentinel2_scaling_factor     
        nir = self.maskCloudCover(scl_band=cloud_20m[0], band_data=nir)     
        swir = self.getRasterData(band=band12_20m[0])
        swir = swir / self.sentinel2_scaling_factor
        swir = self.maskCloudCover(scl_band=cloud_20m[0], band_data=swir)
        
        ndvi_numerator = subtract(nir, r)  
        ndvi_denominator = add(nir, r)
        masked_ndvi = divide(ndvi_numerator, ndvi_denominator)
        masked_ndvi = masked_ndvi.round(decimals=3)
        masked_ndvi = (masked_ndvi * self.output_scale_factor).astype(np.int16)
        reprojected_ndvi = self.createReprojectedGeotiff(source_crs=band4_20m[0], 
                                                         ind_array=masked_ndvi,
                                                         ind_type="NDVI",
                                                         pre_post=pre_post)

        nbr_numerator = subtract(nir, swir)  
        nbr_denominator = add(nir, swir)
        masked_nbr = divide(nbr_numerator, nbr_denominator)   
        masked_nbr = masked_nbr.round(decimals=3)
        masked_nbr = (masked_nbr * self.output_scale_factor).astype(np.int16)
        reprojected_nbr = self.createReprojectedGeotiff(source_crs=band4_20m[0], 
                                                        ind_array=masked_nbr,
                                                        ind_type="NBR",
                                                        pre_post=pre_post)
     
    def run(self):
        os.chdir(self.outdir)
        #if os.path.exists(self.base_dir):
        #    shutil.rmtree(self.base_dir)
        #os.mkdir(self.base_dir)
        
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
            
        self.calculateIndicators(zipped_image=self.pre, pre_post="Pre-fire")
        
        for post_file in self.post:
            self.calculateIndicators(zipped_image=post_file, pre_post="Post-fire")

        os.chdir(self.outdir)
        outputs = os.path.join(self.outdir, self.base_dir, self.region)
        print(f"The outputs are written in:\n{outputs}")



#===================================== RUN ====================================

#################################### Adana ####################################
### Adana Tile 1
pre = "E:/Data/Sentinel-2A/Adana/Adana-pre/S2B_MSIL2A_20210724T081609_N0500_R121_T36SYG_20230219T230606.SAFE.zip"     
post = ["E:/Data/Sentinel-2A/Adana/Adana-post/S2B_MSIL2A_20210803T081609_N0500_R121_T36SYG_20230130T045015.SAFE.zip",
        "E:/Data/Sentinel-2A/Adana/Adana-post/S2B_MSIL2A_20220729T081609_N0400_R121_T36SYG_20220729T100550.SAFE.zip",
        "E:/Data/Sentinel-2A/Adana/Adana-post/S2B_MSIL2A_20230724T081609_N0509_R121_T36SYG_20230724T104606.SAFE.zip",
        "E:/Data/Sentinel-2A/Adana/Adana-post/S2B_MSIL2A_20240728T081609_N0511_R121_T36SYG_20240728T100622.SAFE.zip",
        "E:/Data/Sentinel-2A/Adana/Adana-post/S2B_MSIL2A_20250723T081609_N0511_R121_T36SYG_20250723T110620.SAFE.zip",
        "E:/Data/Sentinel-2A/Adana/Adana-post/S2B_MSIL2A_20260728T081609_N0512_R121_T36SYG_20260728T122227.SAFE.zip"]
shape = {"Area 01": "E:/Data/Sentinel-2A/Shapefiles_revised/Adana_revised/temporal_adana.shp",
         "Area 02": "E:/Data/Sentinel-2A/Shapefiles_revised/Adana_revised/temporal_adana_2.shp",
         "Area 03": "E:/Data/Sentinel-2A/Shapefiles_revised/Adana_revised/temporal_adana_3.shp"}
output_dir = "E:/Data/Indicators_revised"
region_name = "AD"
###############################################################################

ins = Indicator(pre_fire_zip=pre, 
                post_fire_zips_dir=post,
                shapefile_dir=shape,
                output_dir=output_dir,
                region_name=region_name)  # AN, AD, ME, MU

ins.run()