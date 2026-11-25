"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script uses the linear GAM model to examine the contribution of the  #
#    topographical variable (such as aspec, slope, altitude, etc.) on a         #
#    dependent variable (such as Malleability).                                 #
#                                                                               #
#>>> The values of the dependent variable (Y-axis) must be positive. Moreover,  #
#    if aspect is given as an independent variable, it must be in degree's.     #
#                                                                               #
#>>> The script uses 'X_test' and 'y_test' to draw Devience Residual Plots. This#
#    shows the generalization / out-of-sample performance of the model.         #
#                                                                               #
#>>> The ratio of train size/test size must be adjusted through the 'test_size' #
#    argument. For instance, if the argument is adjusted to 0.2, 80% of the data#
#    will be used for training and 20% of the data will be used for testing.    #
#                                                                               #
#>>> Set smoothing parameter (lam) at line 75.                                  #
#                                                                               #
#>>> Set terms at line 76.                                                      #
#                                                                               #
#                             <<< TERM SELECTION >>>                            #
#                                                                               #
#   * Spline Terms (s()):                                                       #
#       Used for modeling non-linear relationships between the feature and the  #
#       dependent variable.                                                     #
#   * Linear Terms (l()):                                                       #
#       Models a linear relationship between the feature and the dependent      #
#       variable.                                                               #
#   * Factor Terms (f()):                                                       #
#       Models categorical or discrete variables with a fixed number of levels. #
#   * Tensor Terms (te()):                                                      #
#       Captures interactions between two or more features, allowing for smooth,#
#       non-linear effects across multiple dimensions. For example the          #
#       interactions of temperature and humidity affecs crop yield.             #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 2.0
year    : 2026

See:
    https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html#Custom-Models
"""

######## spatially-explicit sensitivity analysis

import os
import sys
import pprint
import numpy as np
import pandas as pd
import numpy.ma as ma
import rasterio as rio
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from pygam import LinearGAM, s, l, f, te
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.simplefilter("ignore", UserWarning)

#print(__doc__)

class GAM:
    def __init__(self, independent_variables, 
                 dependent_variable, 
                 aspect_conversion, 
                 area_name, 
                 test_size, 
                 output_dir):
        self.indep_vars = independent_variables
        self.dep_var = dependent_variable
        self.aspect_conv = aspect_conversion
        self.area = area_name
        self.test_size = test_size
        self.outdir = output_dir
        self.mask_val = -1000
        self.lam_grid = np.logspace(-3, 4, 30) # np.logspace(-3, 4, 30)   np.logspace(-3, 3, 10
        self.terms = {"Aspect": s,
                      "Slope": s,
                      "Altitude": s,
                      "Pre-fire NDVI": s, 
                      "Burn Severity": s}  
        self.term_desc = {"s": "Spline",
                          "l": "Linear",
                          "f": "Factor",
                          "te": "Tensor"}
        self.link = {"Poisson GAM": "Log",
                     "Linear GAM": "Identity",
                     "Logistic GAM": "Logit",
                     "Gamma GAM": "Log",
                     "Inv Gauss": "Log"}
        self.distribution = {"Poisson GAM": "Poisson distribution",
                             "Linear GAM": "Normal distribution",
                             "Logistic GAM": "Binomial distribution",
                             "Gamma GAM": "Gamma distribution",
                             "Inv Gauss": "Inv_gauss distribution"}

    def getDataRaster(self, data_name, data_path):   
        scaled_data = ["Burn Severity", "Pre-fire NDVI", "NDVI/Burn", list(self.dep_var.keys())]       
        
        with rio.open(self.dep_var["Recovery Rate"]) as ref_src:   # self.indep_vars["Altitude"]
            ref_arr = ref_src.read(1)
            masked_ref_arr = ma.masked_where(ref_arr == self.mask_val, ref_arr)
            
            with rio.open(data_path) as src:
                data = src.read(1)
                masked_data = np.ma.array(data, mask=masked_ref_arr.mask)
                masked_data = masked_data.flatten()
                masked_data = masked_data.compressed()
                masked_data = masked_data.astype(float)
                masked_data[masked_data==-1000] = np.nan
                       
                if data_name in scaled_data:
                    data_scale_factor = float(src.tags(bidx=0)["SCALE_FACTOR"])
                    masked_data = masked_data * data_scale_factor
        
        return masked_data 
    
    def Degrees2DiscreteDirection(self, degrees):
        directions = ['North', 'Northeast', 'East', 'Southeast', 
                      'South', 'Southwest', 'West', 'Northwest']
        degrees = np.asarray(degrees)
        idx = round(degrees / 45) % 8

        return directions[idx]
    
    def Degrees2NormSinus(self, degrees):
        """
        This function converts the aspect in degrees to normalized sinuosidal 
        pattern. The output raster will contain values in the range of 0 to 1.
        """
        degrees = np.asarray(degrees)
        norm_deg = degrees/359.9999999
        y = (np.sin(2 * np.pi * norm_deg) + 1) / 2
        return y
    
    def MakeDataframe(self):
        data_dict = {}
        for indep_var_name, indep_file in self.indep_vars.items():
            masked_data = self.getDataRaster(data_name=indep_var_name, 
                                             data_path=indep_file)
            data_dict[indep_var_name] = masked_data
            
        for dep_var_name, dep_file in self.dep_var.items():
            masked_data = self.getDataRaster(data_name=dep_var_name, 
                                             data_path=dep_file)
            data_dict[dep_var_name] = masked_data    
        #pprint.pprint(data_dict, sort_dicts=False)
        df = pd.DataFrame(data_dict)
        #print(sum(np.isnan(df["Aspect"])))
        df = df.dropna()
                
        if self.aspect_conv == "normalized_sinusoidal":
            df["Aspect"] = df["Aspect"].apply(self.Degrees2NormSinus)
        elif self.aspect_conv == "discrete_direction":
            df["Aspect"] = df["Aspect"].apply(self.Degrees2DiscreteDirection) 

        return df
        
    def ModelSummary(self, model_type, gam, dep_var, indep_vars, splines, 
                     mae, rmse, pseudo_r2, r2, y_std, y_range, y_mean, dirname):  
        heading = ""
        heading += f"Study Area: {self.area}\n"
        heading += f"Model     : {model_type}\n\n"
        
        heading += f"{'-' * 47} Parameters {'-' * 47}\n"
        heading += f"Dependent variable              : {dep_var}\n" 
        heading += f"Independent variables (features):\n"
        for count, feature in enumerate(indep_vars):
            txt = f"{' ' * 34}Feature ({count}) = {feature}\n"
            heading += str(txt)
        
        heading += f"\n{'-' * 44} Hyperparameters {'-' * 45}\n"
        heading += "Terms and spline types:\n"
        for count, feature in enumerate(indep_vars):
            if self.aspect_conv == "discrete_direction" and feature == "Aspect":
                f_term = self.term_desc["f"]
            else:
                f_term = self.term_desc[self.terms[feature].__name__]
            txt = f"{' ' * 7}Feature ({count}) : Term = {f_term}, Spline type = {splines[feature]}\n"
            heading += str(txt)
        heading += "\nNumber of splines: Is automatically selected.\n"
        heading += "Lambda           : The smoothing parameters (Lambda) are given for each term in the table.\n\n"  
        
        heading += f"{'-' * 42} Model Specifications {'-' * 42}\n"
        heading += f"Link Function: {self.link[model_type]}\n"
        heading += f"Distribution : {self.distribution[model_type]}\n\n"  
        
        heading += f"{'-' * 43} Evaluation Metrics {'-' * 43}\n"
        heading += f"MAE             : {mae}\n"
        heading += f"RMSE            : {rmse}\n"  
        heading += f"RMSE/MAE        : {rmse/mae}\n"   
        heading += f"MAE/Std. Dev.   : {mae/y_std}\n" 
        heading += f"RMSE/Std. Dev.  : {rmse/y_std}\n" 
        heading += f"MAE/Y Range.    : {mae/y_range}\n" 
        heading += f"RMSE/Y Range.   : {rmse/y_range}\n" 
        heading += f"R-Squared       : {r2}\n"
        heading += f"Pseudo R-Squared: {pseudo_r2}\n"
        heading += f"Y (Pred.) Mean  : {y_mean}\n\n"
        
        heading += f"{'-' * 49} Results {'-' * 48}\n"
        
        d_name = model_type.replace(" ", "_")
        final_path = os.path.join(self.outdir, dirname, d_name)
        if not os.path.isdir(final_path):
            os.makedirs(final_path)
        
        output_file = f"{self.area}_gam_summary.txt"
        output_path = os.path.join(final_path, output_file)
        with open(output_path, "w") as file:
            with redirect_stdout(file):
                print(heading)
                print(gam.summary())
                
    def PartialDependencePlot(self, model_type, gam, dep_var, column_names, dirname):        
        plt.figure(figsize=(len(column_names) * 11.7, 6))
        for i, term in enumerate(gam.terms):
            if gam.terms[i].isintercept:  # Skip the intercept term
                continue 
            plt.subplot(1, len(column_names), i + 1)
            XX = gam.generate_X_grid(term=i)                                    
            plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
            plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=0.95)[1], c="r", ls="--")
            plt.xlabel(f"{column_names[i]}", fontsize=38)
            plt.ylabel(dep_var, fontsize=38)        
            
            if self.aspect_conv == "discrete_direction":
                if f"{column_names[i]}" == "Aspect":
                    categories = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                    plt.xticks(ticks=range(8), labels=categories)
            
            plt.yticks(fontsize=32)
            plt.xticks(fontsize=32)
            
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)
            
            xlim = plt.gca().get_xlim()   
            ylim = plt.gca().get_ylim()
            x_padding = 0.17 
            y_padding = 0.05
            x = xlim[0] + x_padding * (xlim[1] - xlim[0])
            y = ylim[1] - y_padding * (ylim[1] - ylim[0])
            plt.text(x, y, f"{self.area}", fontsize=36, 
                     ha="right", va="top", weight="bold",
                     bbox = dict(facecolor = "red", alpha = 0.5))
            plt.tight_layout(pad=1)
        plt.subplots_adjust(hspace=1) 
        
        d_name = model_type.replace(" ", "_")
        fig_name = f"{self.area}_partial_dependence_plots.png"
        final_path = os.path.join(self.outdir, dirname)
        output_path = os.path.join(final_path, d_name, fig_name)
        
        if os.path.exists(output_path):
            os.remove(output_path)
            
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def DrawDevienceResidualsPlot(self, model_type, prediction, devience_residuals, dirname):
        plt.figure(figsize=(8, 6))
        plt.scatter(prediction, devience_residuals, alpha=0.6)
        plt.axhline(0, linestyle="--", color="red")
        plt.xlabel("Predicted Values", fontsize=14)
        plt.ylabel("Deviance Residuals", fontsize=14)
        title = "Deviance Residuals Plot"
        plt.title(title, fontsize=16)
        
        d_name = model_type.replace(" ", "_")
        final_path = os.path.join(self.outdir, dirname)
        fig_name = f"{self.area}_" + title.replace(" ", "_") + ".png"
        output_path = os.path.join(final_path, d_name, fig_name)
                
        if os.path.exists(output_path):
            os.remove(output_path)
            
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    def CalculateIQR(self, data_arr):
        """
        This function calculates the interquartile range (IQR) to remove outliers 
        from dataset using Tukey method.
        """
        q1 = np.percentile(data_arr, 25)
        q3 = np.percentile(data_arr, 75)
        IQR = q3 - q1
        lower_range = q1 - 1.5 * IQR
        upper_range = q3 + 1.5 * IQR
        
        return lower_range, upper_range
            
    def LinearGAM(self, df, Xs_list, dep_var):
        X_df, y_df = df[Xs_list], df[dep_var] 
        X, y = X_df.to_numpy(), y_df.to_numpy()
                
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=self.test_size, 
                                                            random_state=42)
        
        spline_type = {}
        terms = None
        for i in range(len(Xs_list)):
            if self.aspect_conv == "discrete_direction" and Xs_list[i] == "Aspect":
                sp = "cp"
                term = f(i)  
                spline_type[Xs_list[i]] = "Is applicable only Spline term."
            elif self.aspect_conv == "normalized_sinusoidal" and Xs_list[i] == "Aspect":
                if self.terms[Xs_list[i]] == s:
                    sp = "cp"
                    term = self.terms[Xs_list[i]](i, basis=sp)
                    spline_type[Xs_list[i]] = "Cyclic P-splines (cp)"
                else:
                    term = self.terms[Xs_list[i]](i)
                    spline_type[Xs_list[i]] = "Is applicable only Spline term."
            else:
                sp = "ps"
                term = self.terms[Xs_list[i]](i, basis="ps")
                spline_type[Xs_list[i]] = "P-splines (ps)"
                
            if terms is None:
                terms = term
            else:
                terms += term
                
        gam = LinearGAM(terms).gridsearch(X_train, 
                                          y_train, 
                                          lam=self.lam_grid)
        
        y_pred = gam.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred) 
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        pseudo_r2 = gam.statistics_["pseudo_r2"]["explained_deviance"] 
        r2 = r2_score(y_test, y_pred)
        y_std = np.std(y_test)
        y_range = np.max(y_test) - np.min(y_test)
        y_mean = np.nanmean(y_test)
        
        dirname = f"{self.area}_{dep_var}__" + "_".join(Xs_list).replace(" ", "-")
        self.ModelSummary(model_type="Linear GAM", 
                          gam=gam, 
                          dep_var=dep_var, 
                          indep_vars=list(X_df.columns),
                          splines=spline_type,
                          mae=mae,
                          rmse=rmse,
                          pseudo_r2=pseudo_r2,
                          r2=r2,
                          y_std=y_std,
                          y_range=y_range,
                          y_mean=y_mean,
                          dirname=dirname)
        
        self.PartialDependencePlot(model_type="Linear GAM", 
                                   gam=gam, 
                                   dep_var=dep_var, 
                                   column_names=list(X_df.columns),
                                   dirname=dirname)
        
        self.DrawDevienceResidualsPlot(model_type="Linear GAM", 
                                       prediction=gam.predict(X_test), 
                                       devience_residuals=gam.deviance_residuals(X_test, y_test),
                                       dirname=dirname)
    
    def MakeSmmaryCSVs(self, txt_file):
        study_area = None
        model_name = None
        dep_var = None
        indep_vars = ""
        mae = None
        rmse = None
        rmse_mae = None
        mae_std = None
        rmse_std = None
        mae_yrane = None
        rmse_yrane = None
        r_squ = None
        psudo_r_squ = None

        with open(txt_file,"r") as sumfile:
            #file_content = sumfile.read()
            lines = sumfile.readlines()
            for line in lines:
                if line.startswith("Study Area:"):
                    study_area = line.split(":")[-1].strip()
                if line.startswith("Model"):
                    model_name = line.split(":")[-1].strip()
                if line.startswith("Dependent variable"):
                    dep_var = line.split(":")[-1].strip()
                if line.startswith(f"{' ' * 34}Feature"): 
                    ind_var = line.split("=")[-1].strip()
                    if indep_vars == "":
                        indep_vars += ind_var
                    else:
                        indep_vars += f"_{ind_var}"
                if line.startswith("MAE  "):
                    mae = np.round(float(line.split(":")[-1].strip()), decimals=4)
                if line.startswith("RMSE  "):
                    rmse = np.round(float(line.split(":")[-1].strip()), decimals=4)
                if line.startswith("RMSE/MAE"):
                    rmse_mae = np.round(float(line.split(":")[-1].strip()), decimals=4)
                if line.startswith("MAE/Std. Dev."):
                    mae_std = np.round(float(line.split(":")[-1].strip()), decimals=4)
                if line.startswith("RMSE/Std. Dev"):
                    rmse_std = np.round(float(line.split(":")[-1].strip()), decimals=4)
                if line.startswith("MAE/Y Range"):
                    mae_yrane = np.round(float(line.split(":")[-1].strip()), decimals=4)
                if line.startswith("RMSE/Y Range"):
                    rmse_yrane = np.round(float(line.split(":")[-1].strip()), decimals=4)
                if line.startswith("R-Squared"):
                    r_squ = np.round(float(line.split(":")[-1].strip()), decimals=4) 
                if line.startswith("Pseudo R-Squared"):
                    psudo_r_squ = np.round(float(line.split(":")[-1].strip()), decimals=4)
                if line.startswith("Y (Pred.) Mean"):    
                    y_mean = np.round(float(line.split(":")[-1].strip()), decimals=4)
                    
         
        model_data = [study_area, model_name, dep_var, indep_vars, mae, rmse, 
                      rmse_mae, mae_std, rmse_std, mae_yrane, rmse_yrane, r_squ, 
                      psudo_r_squ, y_mean]
            
        return model_data
        
    def Run(self):
        df = self.MakeDataframe()
        df.loc[df["Burn Severity"] < 0, "Burn Severity"] = np.nan
        Xs = ["Aspect", "Slope", "Altitude", "Burn Severity", "Pre-fire NDVI"]
        
        for dep_var, raster in self.dep_var.items():
            if dep_var == "Malleability":
                cdf = df.copy()
                lower_range, upper_range = self.CalculateIQR(cdf["Malleability"].values) 
                lower_range = 0.0
                cdf.loc[~cdf["Malleability"].between(lower_range, upper_range), 
                       "Malleability"] = np.nan     
                cdf[cdf["Malleability"] == 0] = 0.0001 
                
                if self.aspect_conv == "discrete_direction":
                    custom_mapping = {"North": 0, "Northeast": 1, "East": 2, "Southeast": 3,
                                      "South": 4, "Southwest": 5, "West": 6, "Northwest": 7}
                    cdf["Aspect"] = cdf["Aspect"].map(custom_mapping)
                    
                cdf = cdf.dropna()
                for X_var in range(len(Xs)):
                    Xs_list = Xs[0:X_var+1]
                    self.LinearGAM(cdf, Xs_list, dep_var)
                            
            elif dep_var == "Recovery Rate":
                cdf = df.copy()
                lower_range, upper_range = self.CalculateIQR(cdf["Recovery Rate"].values)
                cdf[cdf["Recovery Rate"] < 0] = 0.0001                
                lower_range = 0.0001
                cdf.loc[~cdf["Recovery Rate"].between(lower_range, upper_range), 
                       "Recovery Rate"] = np.nan   

                if self.aspect_conv == "discrete_direction":
                    custom_mapping = {"North": 0, "Northeast": 1, "East": 2, "Southeast": 3,
                                      "South": 4, "Southwest": 5, "West": 6, "Northwest": 7}
                    cdf["Aspect"] = cdf["Aspect"].map(custom_mapping)
                
                cdf = cdf.dropna()    
                for X_var in range(len(Xs)):
                    Xs_list = Xs[0:X_var+1]
                    self.LinearGAM(cdf, Xs_list, dep_var)
                
        col_name = ["Study Area", "Model", "Dependent Var.", "Independent Vars.",
                    "MAE", "RMSE", "RMSE/MAE", "MAE/Std Dev.", "RMSE/Std Dev.",
                    "MAE/Y-Range", "RMSE/Y-Range", "R-Squared", "Pseudo R-Squared",
                    "Mean Predicted Y"]
        summary_df = pd.DataFrame(columns=col_name)        
        
        for root, dirs, files in os.walk(self.outdir):
            for n in files:
                if n.endswith(".txt"):
                    txt_file = os.path.join(root, n)
                    model_data = self.MakeSmmaryCSVs(txt_file)
                    summary_df.loc[len(summary_df)] = model_data
                    area = model_data[0]
                    
        xlsx_filename = f"stat_summary.xlsx"
        xlsx_path = os.path.join(self.outdir, xlsx_filename)           
        summary_df.to_excel(xlsx_path, index=False)
        
        print(f"Outputs saved to:\n{self.outdir}")

    

#===================================== RUN ====================================

#################################### Adana ####################################
"""
### Adana area 1 ###
area_name = "AD01"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/AD01_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/AD01_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/resampled_AD01_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NBR/AD01_dNBR_20210803.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD01_Pre-fire_NDVI_20210724T081609_20m.tif",}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/AD01_malleability_20260728.tif"}   
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/AD01_recovery-percentage_20260728.tif"     
       
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/AD"
"""
"""
### Adana area 2 ###
area_name = "AD02"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/AD02_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/AD02_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/resampled_AD02_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NBR/AD02_dNBR_20210803.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD02_Pre-fire_NDVI_20210724T081609_20m.tif",}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/AD02_malleability_20260728.tif"}    
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/AD02_recovery-percentage_20260728.tif"      
                                                                                     
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/AD"
"""
"""
### Adana area 3 ###
area_name = "AD03"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/AD03_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/AD03_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/resampled_AD03_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NBR/AD03_dNBR_20210803.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD03_Pre-fire_NDVI_20210724T081609_20m.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/AD03_malleability_20260728.tif"}    
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/AD03_recovery-percentage_20260728.tif"  
                                                                                     
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/AD"
"""
###############################################################################

################################### Antalya ###################################
"""
### Antalya area 1 ###
area_name = "AN01"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Antalya/AN01_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Antalya/AN01_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Antalya/resampled_AN01_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AN/NBR/AN01_dNBR_20210814.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AN/NDVI/cropped_AN01_Pre-fire_NDVI_20210720T083559_20m.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AN/NDVI/AN01_malleability_20260729.tif"} 
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AN/NDVI/AN01_recovery-percentage_20260729.tif"   
                                                                                        
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/AN"
"""
"""
### Antalya area 2 ###
area_name = "AN02"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Antalya/AN02_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Antalya/AN02_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Antalya/resampled_AN02_masked_DEM.tif",  
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AN/NBR/AN02_dNBR_20210824.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AN/NDVI/cropped_AN02_Pre-fire_NDVI_20210720T083559_20m.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AN/NDVI/AN02_malleability_20260729.tif"}     
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AN/NDVI/AN02_recovery-percentage_20260729.tif"
                                                                                                    
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/AN"
"""
###############################################################################

################################### Mersin ####################################
"""
### Mersin area 1 ###
area_name = "ME01"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mersin/ME01_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mersin/ME01_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mersin/resampled_ME01_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/ME/NBR/ME01_dNBR_20210811.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/ME/NDVI/cropped_ME01_Pre-fire_NDVI_20210702T082601_20m.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/ME/NDVI/ME01_malleability_20260721.tif"}   
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/ME/NDVI/ME01_recovery-percentage_20260721.tif"

out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/ME"
"""
"""
### Mersin area 2 ###
area_name = "ME02"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mersin/ME02_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mersin/ME02_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mersin/resampled_ME02_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/ME/NBR/ME02_dNBR_20210811.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/ME/NDVI/cropped_ME02_Pre-fire_NDVI_20210702T082601_20m.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/ME/NDVI/ME02_malleability_20260721.tif"}   
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/ME/NDVI/ME02_recovery-percentage_20260721.tif" 
                                                                                     
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/ME"
"""
###############################################################################

#################################### Mugla ####################################
"""
### Mugla area 1 ###
area_name = "MU01"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/MU01_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/MU01_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/resampled_MU01_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NBR/MU01_dNBR_20210810.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/cropped_MU01_Pre-fire_NDVI_20210716T085559_20m.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/MU01_malleability_20260727.tif"}  
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/MU01_recovery-percentage_20260727.tif"  
                                                                                               
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/MU"
"""
"""
### Mugla area 2 ###
area_name = "MU02"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/MU02_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/MU02_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/resampled_MU02_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NBR/MU02_dNBR_20210817.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/cropped_MU02_Pre-fire_NDVI_20210718T084601_20m.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/MU02_malleability_20260727.tif"}    
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/MU02_recovery-percentage_20260727.tif"
                                                                                                 
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/MU"
"""
"""
### Mugla area 3 ###
area_name = "MU03"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/MU03_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/MU03_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/resampled_MU03_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NBR/MU03_dNBR_20210817.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/cropped_MU03_Pre-fire_NDVI_20210718T084601_20m.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/MU03_malleability_20260727.tif"}
          #"Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/MU03_recovery-percentage_20260727.tif"
          
#dep_var = {}
                                                                                                    
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/MU"
"""

### Mugla area 4
area_name = "MU04"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/MU04_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/MU04_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Mugla/resampled_MU04_masked_DEM.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NBR/MU04_dNBR_20210812.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/cropped_MU04_Pre-fire_NDVI_20210728T084601_20m.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/MU04_malleability_20260727.tif"}
          # "Recovery Rate": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/MU/NDVI/MU04_recovery-percentage_20260727.tif"
          
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/MU"


###############################################################################

ins = GAM(independent_variables=indep_vars,
                  dependent_variable=dep_var,
                  aspect_conversion="normalized_sinusoidal",   # ["normalized_sinusoidal", "discrete_direction"]
                  area_name=area_name,
                  test_size=0.2,             # 80% for train 20% for test
                  output_dir=out_path)

ins.Run()
