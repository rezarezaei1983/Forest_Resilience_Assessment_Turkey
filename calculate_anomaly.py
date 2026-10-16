"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                               <<< SOME NOTES >>>                              #
#                                                                               #
#>>> This script reads output from the                                          #
#    'ERA5_get_seasonal_stat_for_precipitation.py' and                          #
#    'ERA5_get_seasonal_stat_for_temperature.py' scrips to calculate the        #
#    seasonal anomaly in meteoroloical variable of the post-fire period based   #
#    on the historical period data. The calculations are conduced for seasonal  #
#    and yearly/seasonal periods.                                               # 
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
import pandas as pd


class CalculateAnomaly():
    def __init__(self, input_meteo_files, output_dir, pre_fire_st_end_date,
                 post_fire_st_end_date):
        self.input_csv = input_meteo_files
        self.outdir = output_dir
        self.hist_period = pre_fire_st_end_date
        self.curr_period = post_fire_st_end_date
        
    def getSeason(self, date):
        Y = 2000  # Replace all years to 2000
        date_fixed = date.replace(year=Y)
        if pd.Timestamp(f"{Y}-03-01") <= date_fixed <= pd.Timestamp(f"{Y}-05-31"):
            return "Spring"
        elif pd.Timestamp(f"{Y}-06-01") <= date_fixed <= pd.Timestamp(f"{Y}-08-31"):
            return "Summer"
        elif pd.Timestamp(f"{Y}-09-01") <= date_fixed <= pd.Timestamp(f"{Y}-11-30"):
            return "Autumn"
        else:
            return "Winter"
    
    def makeDataFrame(self, index):
        idx = [i for i in index]
        df = pd.DataFrame(index=idx,
                          columns=["Anomaly"])
        return df
        
    def makeCSV(self):
        season_list = ["Spring", "Summer", "Autumn", "Winter"]
        
        for area, csv_file in self.input_csv.items():
            file_name = os.path.basename(csv_file)
            data_name = file_name[16:-4].replace("_", " ")
            input_csv = pd.read_csv(csv_file)
            input_csv["Date"] = pd.to_datetime(input_csv["Date"])
        
            pre_fire_st = datetime.datetime.strptime(self.hist_period[0], "%Y-%m-%d")
            pre_fire_ed = datetime.datetime.strptime(self.hist_period[1], "%Y-%m-%d")
            pre_fire_df = input_csv[(input_csv["Date"] >= pre_fire_st) & 
                                    (input_csv["Date"] <= pre_fire_ed)].copy()
        
            post_fire_st = datetime.datetime.strptime(self.curr_period[0], "%Y-%m-%d")
            post_fire_ed = datetime.datetime.strptime(self.curr_period[1], "%Y-%m-%d")
            post_fire_df = input_csv[(input_csv["Date"] >= post_fire_st) & 
                                     (input_csv["Date"] <= post_fire_ed)].copy()

            pre_fire_df.loc[:, "season"] = pre_fire_df["Date"].apply(self.getSeason)
            pre_fire_season_avg = pre_fire_df.groupby("season")[data_name].mean().reindex(season_list)   
            pre_fire_season_avg_df = pre_fire_season_avg.to_frame(name="mean")
            
            post_fire_df.loc[:, "season"] = post_fire_df["Date"].apply(self.getSeason)
            post_fire_df.loc[:, "Year"] = post_fire_df["Date"].dt.year
            
            post_fire_season_avg = post_fire_df.groupby("season")[data_name].mean().reindex(season_list)
            post_fire_season_avg = post_fire_season_avg.to_frame(name="mean")
            
            seasonal_anomaly = self.makeDataFrame(index=season_list)
            for index, row in post_fire_season_avg.iterrows():
                post_cur_season_avg = row["mean"]
                pre_cur_season_avg = pre_fire_season_avg_df["mean"].loc[index]
                anomaly = round((post_cur_season_avg - pre_cur_season_avg), 6)
                seasonal_anomaly.loc[index, "Anomaly"] = anomaly
                
            post_fire_year_season_avg = (post_fire_df.groupby(["Year", "season"])[data_name].agg(["mean"]).reset_index())
            year_list = list(post_fire_year_season_avg["Year"].unique())

            year_season_idx = []
            for year in year_list:
                for season in season_list:
                    year_season = f"{year} {season}"
                    year_season_idx.append(year_season)
            
            yearly_seasonal_anomaly = self.makeDataFrame(index=year_season_idx)
            for index, row in post_fire_year_season_avg.iterrows():
                #print(row["Year"], row["season"], row["mean"])
                post_cur_year_season_avg = row["mean"]
                pre_cur_season_avg = pre_fire_season_avg_df["mean"].loc[row["season"]]
                anomaly = round((post_cur_year_season_avg - pre_cur_season_avg), 6)
                yearly_seasonal_anomaly.loc[f"{row['Year']} {row['season']}", "Anomaly"] = anomaly
            
            csv_name = f"anomaly_of_{file_name[16:-4]}_{area}.csv" 
            csv_filename = os.path.join(self.outdir, csv_name) 
            
            with open(csv_filename, "w", encoding="utf-8") as f:
                f.write(f"Study Area: {area}\n")
                f.write(f"Variable: {data_name}\n\n")
                f.write("Post-fire period anomaly by only season:\n")
                
            seasonal_anomaly.to_csv(csv_filename, mode="a", index=True)

            with open(csv_filename, "a", encoding="utf-8") as f:
                f.write("\n\n\nPost-fire period anomaly by year/season:\n")

            yearly_seasonal_anomaly.to_csv(csv_filename, mode="a", index=True)
  
        print(f"The outputs area saved in:\n{self.outdir}")

#===================================== RUN ====================================
"""
############################### Mean Temperature ##############################
daily_meteo = {"AD01":"E:/Data/ERA5/out/AD01_average_of_Mean_2m_Temperature.csv",
               "AD02":"E:/Data/ERA5/out/AD02_average_of_Mean_2m_Temperature.csv",
               "AD03":"E:/Data/ERA5/out/AD03_average_of_Mean_2m_Temperature.csv",
               "AN01":"E:/Data/ERA5/out/AN01_average_of_Mean_2m_Temperature.csv",
               "AN02":"E:/Data/ERA5/out/AN02_average_of_Mean_2m_Temperature.csv",
               "ME01":"E:/Data/ERA5/out/ME01_average_of_Mean_2m_Temperature.csv",
               "ME02":"E:/Data/ERA5/out/ME02_average_of_Mean_2m_Temperature.csv",
               "MU01":"E:/Data/ERA5/out/MU01_average_of_Mean_2m_Temperature.csv",
               "MU02":"E:/Data/ERA5/out/MU02_average_of_Mean_2m_Temperature.csv",
               "MU03":"E:/Data/ERA5/out/MU03_average_of_Mean_2m_Temperature.csv",
               "MU04":"E:/Data/ERA5/out/MU04_average_of_Mean_2m_Temperature.csv"}
output_dir = "E:/Data/ERA5/Anomaly"
###############################################################################
"""

############################# Total Precipitation #############################
daily_meteo = {"AD01":"E:/Data/ERA5/out/AD01_average_of_Total_Precipitation.csv",
               "AD02":"E:/Data/ERA5/out/AD02_average_of_Total_Precipitation.csv",
               "AD03":"E:/Data/ERA5/out/AD03_average_of_Total_Precipitation.csv",
               "AN01":"E:/Data/ERA5/out/AN01_average_of_Total_Precipitation.csv",
               "AN02":"E:/Data/ERA5/out/AN02_average_of_Total_Precipitation.csv",
               "ME01":"E:/Data/ERA5/out/ME01_average_of_Total_Precipitation.csv",
               "ME02":"E:/Data/ERA5/out/ME02_average_of_Total_Precipitation.csv",
               "MU01":"E:/Data/ERA5/out/MU01_average_of_Total_Precipitation.csv",
               "MU02":"E:/Data/ERA5/out/MU02_average_of_Total_Precipitation.csv",
               "MU03":"E:/Data/ERA5/out/MU03_average_of_Total_Precipitation.csv",
               "MU04":"E:/Data/ERA5/out/MU04_average_of_Total_Precipitation.csv"}
output_dir = "E:/Data/ERA5/Anomaly"
###############################################################################


ins = CalculateAnomaly(input_meteo_files=daily_meteo,        
                       output_dir=output_dir,
                       pre_fire_st_end_date=["1990-01-01", "2021-02-28"],       
                       post_fire_st_end_date=["2021-09-01", "2026-07-31"])

ins.makeCSV()

