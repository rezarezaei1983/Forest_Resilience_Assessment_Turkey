"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                               <<< SOME NOTES >>>                              #
#                                                                               #
#>>> This script reads output from 'ERA5_data_pickle.py' and gets the seasonal  #
#    average temperature for each year within the shapefiles.                   #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2026
"""

import os
import sys
import csv
import pprint
import datetime
import numpy as np
import pandas as pd
import _pickle as cPickle
import geopandas as gpd
from shapely.geometry import mapping
from rasterio.features import rasterize
from rasterio.transform import from_origin

class ReadPKL:
    def __init__(self, input_pkl, shapefile, output_dir):
        self.pkl = input_pkl
        self.shape = shapefile
        self.outdir = output_dir
                                                          
    def cropDataWithShapefile(self, data, lat, lon, shapefile):
        gdf = gpd.read_file(shapefile)
        geometries = [mapping(geom) for geom in gdf.geometry]
        nrows, ncols = data.shape
        res_y = abs(lat[1, 0] - lat[0, 0])
        res_x = abs(lon[0, 1] - lon[0, 0])
        transform = from_origin(lon[0, 0] - res_x / 2, 
                                lat[0, 0] - res_y / 2,
                                res_x, 
                                res_y)

        mask = rasterize(geometries, out_shape=(nrows, ncols), transform=transform,
                         fill=0, all_touched=True, dtype=np.uint8)
        cropped = np.where(mask == 1, data, np.nan)
        
        return cropped
    
    def getSeason(self, date):
        Y = 2000  
        date_fixed = date.replace(year=Y)
        if pd.Timestamp(f"{Y}-03-01") <= date_fixed <= pd.Timestamp(f"{Y}-05-31"):
            return "Spring"
        elif pd.Timestamp(f"{Y}-06-01") <= date_fixed <= pd.Timestamp(f"{Y}-08-31"):
            return "Summer"
        elif pd.Timestamp(f"{Y}-09-01") <= date_fixed <= pd.Timestamp(f"{Y}-11-30"):
            return "Autumn"
        else:
            return "Winter"
        
    def readPKL(self):
        for area_name, shapefile in self.shape.items():
            file_name = os.path.basename(self.pkl)
            data_name = file_name[:-4].replace("_", " ")
            timeseries_csv_name = f"{area_name}_average_of_{file_name[:-4]}.csv" 
            
            data_dict = {}
            with open(self.pkl, "rb") as fp:
                pkld_dt = cPickle.load(fp)
                lat = pkld_dt["Lat"]
                lon = pkld_dt["Lon"]
                for year, year_dict in pkld_dt.items():
                    if type(year_dict) == dict:
                       for day, day_data in year_dict.items():
                           cropped_data = self.cropDataWithShapefile(data=day_data, 
                                                                     lat=lat, 
                                                                     lon=lon, 
                                                                     shapefile=shapefile)
                           mean_val = np.nanmean(cropped_data)
                           mean_val = mean_val - 273.15
                           data_dict[day] = mean_val
            
            df = pd.DataFrame(list(data_dict.items()), columns=["Date", data_name])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values(by="Date")
            
            csv_filename = os.path.join(self.outdir, timeseries_csv_name)
            df.to_csv(csv_filename, index=False)
            df["Date"] = pd.to_datetime(df["Date"], format="mixed")
            df["Season"] = df["Date"].map(self.getSeason)
            df["Year"] = df["Date"].dt.year
            
            seasonal_stats = (df.groupby(["Year", "Season"])[data_name].agg(["mean"]).reset_index())
            yr_season_avg_csv_name = f"{area_name}_yearly_seasonal_average_{file_name[:-4]}.csv" 
            avg_csv_filename = os.path.join(self.outdir, yr_season_avg_csv_name)  
            seasonal_stats.to_csv(avg_csv_filename, index=True) 


#===================================== RUN ====================================

input_pkl = "E:/Data/ERA5/Mean_2m_Temperature.pkl"

shapefile = {"AD01":"E:/Data/Sentinel-2A/Shapefiles_Revised/Adana_revised/temporal_adana.shp",
             "AD02":"E:/Data/Sentinel-2A/Shapefiles_Revised/Adana_revised/temporal_adana_2.shp",
             "AD03":"E:/Data/Sentinel-2A/Shapefiles_Revised/Adana_revised/temporal_adana_3.shp",
             "AN01":"E:/Data/Sentinel-2A/Shapefiles_Revised/Antalya_revised/area_01.shp",
             "AN02":"E:/Data/Sentinel-2A/Shapefiles_Revised/Antalya_revised/area_02.shp",
             "ME01":"E:/Data/Sentinel-2A/Shapefiles_Revised/Mersin_revised/mersin_area_01.shp",
             "ME02":"E:/Data/Sentinel-2A/Shapefiles_Revised/Mersin_revised/mersin_area_02.shp",
             "MU01":"E:/Data/Sentinel-2A/Shapefiles_Revised/Mugla_revised/area_1.shp",
             "MU02":"E:/Data/Sentinel-2A/Shapefiles_Revised/Mugla_revised/area_2.shp",
             "MU03":"E:/Data/Sentinel-2A/Shapefiles_Revised/Mugla_revised/area_3.shp",
             "MU04":"E:/Data/Sentinel-2A/Shapefiles_Revised/Mugla_revised/area_4.shp"}

output_dir="E:/Data/ERA5/out"

ins = ReadPKL(input_pkl=input_pkl,        
              shapefile=shapefile,
              output_dir=output_dir) 

ins.readPKL()