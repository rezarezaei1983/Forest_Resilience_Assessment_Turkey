
# Data source:
#    https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=download    

import os
import sys
import glob
import pprint
import numpy as np
import _pickle as cPickle
from netCDF4 import Dataset
from datetime import datetime, timedelta


class ERA5:
    def __init__(self, input_dir, output_dir, variable_name, variable_description):
        self.indir = input_dir
        self.outdir = output_dir
        self.var_name = variable_name
        self.var_desc = variable_description
        self.file_name = "*.nc"
        
    def getDates(self):
        if os.path.exists(self.indir):       
            files = os.path.join(self.indir, self.file_name) 
            infiles = glob.glob(files)
        else:
            print("Error: Input files directory not found.")
            sys.exit()
        
        return infiles
    
    def getLatLon(self, infile):        
        nc_file = Dataset(infile, "r")                                               
        lat = nc_file.variables["latitude"][:].data
        lon = nc_file.variables["longitude"][:].data
        lon2d, lat2d = np.meshgrid(lon, lat)
        
        return lon2d, lat2d
    
    def getDate(self, start_date, increment):
        date = start_date + timedelta(days=increment)
        date = date.strftime("%Y-%m-%d")
        
        return date
        
    def makeDataDict(self):
        infiles = self.getDates()
        lon2d, lat2d = self.getLatLon(infile=infiles[0])

        data_dict = {}
        for file in infiles:
            nc_file = Dataset(file, "r")  
            target = nc_file.variables[self.var_name][:,:,:].data      # (valid_time, latitude, longitude)
            t_unit = nc_file.variables["valid_time"].units 
            start_date = datetime.strptime(t_unit.split(" ")[-2], "%Y-%m-%d")
            year = start_date.year 

            if year not in data_dict.keys():
                data_dict[year] = {}
                
            for count, days in enumerate(range(len(target)), start=0):
                curr_date = self.getDate(start_date=start_date, increment=count)
                dt = nc_file.variables[self.var_name][days,:,:].data
                data_dict[year][curr_date] = dt
                        
        sorted_dict = dict(sorted(data_dict.items()))
                    
        sorted_dict["Lat"] = lat2d
        sorted_dict["Lon"] = lon2d
        #print(sorted_dict.keys())
        #print(data_dict[2026].keys())
        
        pkl_name = self.var_desc.replace(" ", "_")
        pkl_name = f"{pkl_name}.pkl"
        out_pkl_name = os.path.join(self.outdir, pkl_name)
        
        with open(out_pkl_name, "wb") as fp:
            cPickle.dump(sorted_dict, fp)
        
        print(f"Outputs saved to:\n{out_pkl_name}")




#===================================== RUN ====================================
input_dir = "E:/Data/ERA5/ERA5_Max_2m_Temp"
#input_dir = "E:/Data/ERA5/ERA5_Mean_2m_Temp"
#input_dir = "E:/Data/ERA5/ERA5_Total_Precipitation"

ins = ERA5(input_dir=input_dir,
           output_dir="E:/Data/ERA5",
           variable_name="t2m",    # total prec.: "tp",    temp: "t2m" 
           variable_description="Maximum 2m Temperature")  # "Maximum 2m Temperature", "Mean 2m Temperature", "Total Precipitation"

ins.makeDataDict()