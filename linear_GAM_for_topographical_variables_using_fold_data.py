"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script uses the Linear GAM model to examine the contribution of the  #
#    topographical variable (such as aspec, slope, altitude, etc.) on a         #
#    dependent variable (such as Malleability).                                 #
#                                                                               #
#>>> The values of the dependent variable (Y-axis) must be positive. Moreover,  #
#    if aspect is given as an independent variable, it must be in degree's.     #
#                                                                               #
#>>> The script uses 'X_test' and 'y_test' to draw Devience Residual Plots. This#
#    shows the generalization / out-of-sample performance of the model.         #
#                                                                               #
#>>> The code creates 5 folds to assess spatial atocorrelation. The model       #
#    predicts the out-of-fold response variable.                                #
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
version : 3.0
year    : 2026

See:
    https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html#Custom-Models
"""


import os
import sys
import shutil
import pprint
import numpy as np
import pandas as pd
import numpy.ma as ma
import rasterio as rio
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from contextlib import redirect_stdout
from pygam import LinearGAM, s, l, f, te
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skgstat import Variogram
from libpysal.weights import DistanceBand
from esda import Moran

import warnings
warnings.simplefilter("ignore", UserWarning)

#print(__doc__)


class GAM:
    def __init__(self, independent_variables, dependent_variable, aspect_conversion, 
                 area_name, output_dir):
        self.indep_vars = independent_variables
        self.dep_var = dependent_variable
        self.aspect_conv = aspect_conversion
        self.area = area_name
        self.outdir = output_dir
        self.mask_val = -1000
        self.lam_grid = np.logspace(-3, 4, 30)
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
    
    def selectUTMzone(self, central_lon, central_lat):
        hemisphere = "N" if central_lat >= 0 else "S"
        utm_zone = int((central_lon + 180) // 6) + 1
        
        # Define destination CRS
        epsg_code = (32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone)
        dst_crs = f"EPSG:{epsg_code}"
        
        return dst_crs
        
    def reprojectRaster(self, input_file):
        """
        Reproject to get coordinates in meters.
        """
        with rio.open(input_file) as src:
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        
            west, south = transformer.transform(src.bounds.left, src.bounds.bottom)
            east, north = transformer.transform(src.bounds.right, src.bounds.top)
            central_lon = (west + east) / 2
            central_lat = (south + north) / 2
            dst_crs = self.selectUTMzone(central_lon, central_lat)
            
            # Preserve original dimensions
            width = src.width
            height = src.height
            
            transform, _, _ = calculate_default_transform(src.crs,
                                                          dst_crs,
                                                          width,
                                                          height,
                                                          *src.bounds,
                                                          dst_width=width,
                                                          dst_height=height)
            
            kwargs = src.meta.copy()
            kwargs.update({"crs": dst_crs,
                           "transform": transform,
                           "width": width,
                           "height": height,
                           "nodata": self.mask_val})
            
            file_path = os.path.dirname(input_file)
            file_name = f"reprojected_{os.path.basename(input_file)}"
            reprojected_file_path = os.path.join(file_path, file_name)
            
            # Reproject
            with rio.open(reprojected_file_path, "w", **kwargs) as dst:
                for band in range(1, src.count + 1):
                    # Initialize destination array with NoData
                    destination = np.full((height, width),
                                           self.mask_val,
                                           dtype=src.dtypes[band - 1])
                    
                    reproject(source=rio.band(src, band),
                              destination=destination,
                              src_transform=src.transform,
                              src_crs=src.crs,
                              src_nodata=self.mask_val,
                              dst_transform=transform,
                              dst_crs=dst_crs,
                              dst_nodata=self.mask_val,
                              resampling=Resampling.nearest)
                    dst.update_tags(**src.tags())
                    dst.write(destination, band)
                    
                    dest_pxl_size_x, dest_pxl_size_y = dst.res
            
            return reprojected_file_path, dest_pxl_size_x, dest_pxl_size_y
        
    def getDataRaster(self, data_name, data_path):   
        scaled_data = ["Burn Severity", "Pre-fire NDVI", "NDVI/Burn", list(self.dep_var.keys())]       
        
        with rio.open(self.dep_var["Malleability"]) as ref_src:
            ref_arr = ref_src.read(1)
            masked_ref_arr = ma.masked_where(ref_arr == self.mask_val, ref_arr)
            
            with rio.open(data_path) as src:
                data = src.read(1)
                masked_data = np.ma.array(data, mask=masked_ref_arr.mask)                
                masked_data = masked_data.flatten()
                masked_data = masked_data.compressed()
                masked_data = masked_data.astype(float)
                masked_data[masked_data==self.mask_val] = np.nan
                                
                if data_name in scaled_data:
                    data_scale_factor = float(src.tags(bidx=0)["SCALE_FACTOR"])
                    masked_data = masked_data * data_scale_factor
                                                   
        return masked_data 
    
    def getCoordinates(self, data_path):
        with rio.open(self.dep_var["Malleability"]) as ref_src:   # self.dep_var["Recovery Rate"]
            ref_arr = ref_src.read(1)
            masked_ref_arr = ma.masked_where(ref_arr == self.mask_val, ref_arr)
        
        with rio.open(data_path) as src:
            rows, cols = np.indices((src.height, src.width))
            X, Y = rio.transform.xy(src.transform,
                                    rows,
                                    cols,
                                    offset="center")
            X = np.array(X)
            Y = np.array(Y)
            X_masked = ma.array(X, mask=masked_ref_arr.mask)
            Y_masked = ma.array(Y, mask=masked_ref_arr.mask)
            X_flat = X_masked.compressed()
            Y_flat = Y_masked.compressed()
        
        return X_flat, Y_flat
        
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
            reprojected_file_path, dest_pxl_size_x, dest_pxl_size_y = self.reprojectRaster(input_file=indep_file)
            X_flat, Y_flat = self.getCoordinates(data_path=reprojected_file_path)
            masked_data = self.getDataRaster(data_name=indep_var_name, 
                                             data_path=reprojected_file_path)
                        
            if "Longitude" not in data_dict.keys():
                data_dict["Longitude"] = X_flat
                data_dict["Latitude"] = Y_flat
            
            data_dict[indep_var_name] = masked_data
            
        for dep_var_name, dep_file in self.dep_var.items():
            
            reprojected_file_path, dest_pxl_size_x, dest_pxl_size_y = self.reprojectRaster(input_file=dep_file)
            masked_data = self.getDataRaster(data_name=dep_var_name, 
                                             data_path=reprojected_file_path)
            data_dict[dep_var_name] = masked_data    
            
        #pprint.pprint(data_dict, sort_dicts=False)
        df = pd.DataFrame(data_dict)
        #print(sum(np.isnan(df["Aspect"])))
        df = df.dropna()
               
        if self.aspect_conv == "normalized_sinusoidal":
            df["Aspect"] = df["Aspect"].apply(self.Degrees2NormSinus)
        elif self.aspect_conv == "discrete_direction":
            df["Aspect"] = df["Aspect"].apply(self.Degrees2DiscreteDirection) 
        
        return df, dest_pxl_size_x, dest_pxl_size_y
                        
    def PartialDependencePlot(self, model_type, gam, dep_var, column_names, dirname, fold_num):        
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
        fig_name = f"{self.area}_fold-{fold_num}_{dep_var}_partial_dependence_plots.png"
        output_path = os.path.join(dirname, fig_name)
            
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def DrawDevienceResidualsPlot(self, model_type, prediction, devience_residuals, dirname, fold_num):
        plt.figure(figsize=(8, 6))
        plt.scatter(prediction, devience_residuals, alpha=0.6)
        plt.axhline(0, linestyle="--", color="red")
        plt.xlabel("Predicted Values", fontsize=14)
        plt.ylabel("Deviance Residuals", fontsize=14)
        title = "Deviance Residuals Plot"
        plt.title(title, fontsize=16)
        
        fig_name = f"{self.area}_fold-{fold_num}_" + title.replace(" ", "-") + ".png"
        output_path = os.path.join(dirname, fig_name)
                
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    def ModelSummary(self, model_type, gam, dep_var, indep_vars, splines, 
                     mae, rmse, pseudo_r2, r2, y_range, y_mean, dirname, fold_num):  
        heading = ""
        heading += f"Study Area: {self.area}\n"
        heading += f"Model     : {model_type}\n"
        heading += f"Fold      : {fold_num}\n\n"
        
        heading += f"{'-' * 47} Parameters {'-' * 47}\n"
        heading += f"Dependent variable              : {dep_var}\n" 
        heading += "Independent variables (features):\n"
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
        heading += f"MAE/Y Range.    : {mae/y_range}\n" 
        heading += f"RMSE/Y Range.   : {rmse/y_range}\n" 
        heading += f"R-Squared       : {r2}\n"
        heading += f"Pseudo R-Squared: {pseudo_r2}\n"
        heading += f"Y (Pred.) Mean  : {y_mean}\n\n"
        
        heading += f"{'-' * 49} Results {'-' * 48}\n"
                
        output_file = f"{self.area}_fold-{fold_num}_gam-summary.txt"
        output_path = os.path.join(dirname, output_file)
        with open(output_path, "w") as file:
            with redirect_stdout(file):
                print(heading)
                print(gam.summary())
        
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
        
    def EstimateVariogram(self, df, dep_var, out_dir, max_samples=3000,
                      n_lags=15, model="spherical", random_state=42, plot=True):

        spatial_df = df[["Longitude", "Latitude", dep_var]].copy()
        spatial_df = spatial_df.replace([np.inf, -np.inf], np.nan).dropna()
        coords = spatial_df[["Longitude", "Latitude"]].to_numpy(dtype=float)
        values = spatial_df[dep_var].to_numpy(dtype=float)

        if len(values) < 100:
            raise ValueError("Insufficient valid observations for variogram analysis.")

        # Select a representative sample
        rng = np.random.default_rng(random_state)
        sample_size = min(max_samples, len(values))
    
        if sample_size < len(values):
            sample_idx = rng.choice(len(values),
                                    size=sample_size,
                                    replace=False)
            coords_sample = coords[sample_idx]
            values_sample = values[sample_idx]
        else:
            coords_sample = coords
            values_sample = values
            
        # Calculate empirical and fitted variogram   
        V = Variogram(coordinates=coords_sample,
                      values=values_sample,
                      model=model,
                      n_lags=n_lags,
                      normalize=False)
    
        parameters = V.parameters
        effective_range = parameters[0]
        sill = parameters[1]
        nugget = parameters[2]
                
        if plot:
            fig = V.plot()
            fig.suptitle(f"Variogram of {dep_var} ({model} model)")
            ax = fig.axes[0]
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Semivariance")
            
            filename = f"{self.area}_variogram_of_{dep_var}_{model}-model.png"
            path = os.path.join(out_dir, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
                            
        self.variogram = V
        self.spatial_autocorrelation_range = effective_range
    
        variogram_results = {"variogram": V,
                             "model": model,
                             "effective_range": effective_range,
                             "sill": sill,
                             "nugget": nugget,
                             "sample_size": sample_size}
        
        return variogram_results
    
    def CalculateResidualMoran(self, coords, residuals, fold, threshold, permutations=499):  # 499
        # Remove invalid observations
        valid_mask = (np.isfinite(coords).all(axis=1) & np.isfinite(residuals))
    
        coords = coords[valid_mask]
        residuals = residuals[valid_mask]
            
        if len(residuals) < 3:
            raise ValueError(f"Insufficient observations in fold {fold} "
                             "to calculate Moran's I.")
                
        # Construct spatial weights using a distance threshold.
        # Pixels within the threshold are considered neighbors.
        weights = DistanceBand.from_array(coords,
                                          threshold=threshold,
                                          binary=True,
                                          silence_warnings=True)
        
        # Count observations without neighbors
        n_islands = len(weights.islands)
        island_percentage = (len(weights.islands) / len(residuals)) * 100
        
        # Row-standardize the spatial weights.
        weights.transform = "R"
    
        # Calculate Global Moran's I.
        moran = Moran(residuals, weights, permutations=permutations)
                
        results = {"Fold": fold,
                   "N_test": len(residuals),
                   "N_islands": n_islands,
                   "Island_percentage": island_percentage,
                   "Moran_I": moran.I,
                   "Expected_I": moran.EI,
                   "Z_score": moran.z_sim,
                   "P_value": moran.p_sim}
    
        return results
                
    def PlotResidualMap(self, residuals_df, map_extent, dirname, fold_num):    
        # Extract coordinates and residuals                            
        x = residuals_df["X"].to_numpy()
        y = residuals_df["Y"].to_numpy()
        residuals = residuals_df["Residual"].to_numpy()
    
        # Symmetric colour scale around zero
        max_abs = np.nanmax(np.abs(residuals))
    
        if not np.isfinite(max_abs) or max_abs == 0:
            raise ValueError("Residuals must contain non-zero finite values.")
    
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(x,
                             y,
                             c=residuals,
                             cmap="RdBu_r",
                             norm=norm,
                             s=5,
                             edgecolors="none")
        
        ax.set_aspect("equal")
        ax.grid(False)
        
        plt.tight_layout()
                
        fig.canvas.draw()
        #ax_pos = ax.get_position()
        cax = fig.add_axes([ax.get_position().x1 + 0.02,
                            ax.get_position().y0,0.02,
                            ax.get_position().height])
        
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Residual (Observed - Predicted)", fontsize=14)
    
        ax.set_xlabel("X coordinate (m)", fontsize=14)
        ax.set_ylabel("Y coordinate (m)", fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        ax.set_title(f"Spatial Distribution of CV Residuals (Fold={fold_num})", fontsize=16)
        
        #ax.set_xlim(map_extent[0], map_extent[1])
        #ax.set_ylim(map_extent[2], map_extent[3])
        
        fig_name = f"Residual_Map_Fold_{fold_num}.png"
        output_path = os.path.join(dirname, fig_name)
        
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
        plt.show()
        plt.close(fig)   
    
    def BuildGAMTerms(self, Xs_list):
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
                
        return terms, spline_type
        
    def LinearGAM(self, df, Xs_list, dep_var, pixel_size_x, pixel_size_y):
        xs = "_".join(Xs_list).replace(" ", "-") 
        output_dir = f"{self.area}_{dep_var}__{xs}"
        final_path = os.path.join(self.outdir, output_dir)
        
        # The distance between diagonally adjacent pixel centres
        neighbor_distance = (np.sqrt(pixel_size_x**2 + pixel_size_y**2) * 1.01)
        
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        
        os.makedirs(final_path)
                
        X_df, y_df = df[Xs_list], df[dep_var] 
        X, y = X_df.to_numpy(dtype=float), y_df.to_numpy(dtype=float)
        coords = df[["Longitude", "Latitude"]].to_numpy(dtype=float)
        
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        #map_extent = (x_min, x_max, y_min, y_max)
        
        # Add 2% padding
        x_padding = (x_max - x_min) * 0.02
        y_padding = (y_max - y_min) * 0.02
        
        map_extent = (x_min - x_padding,
                      x_max + x_padding,
                      y_min - y_padding,
                      y_max + y_padding)
        #print(map_extent)
        #sys.exit()
        
        variogram_results = self.EstimateVariogram(df=df,
                                                   dep_var=dep_var,
                                                   out_dir=final_path, 
                                                   max_samples=3000,
                                                   n_lags=15,
                                                   model="spherical")
        
        autocorr_range = variogram_results["effective_range"]
        
        if not np.isfinite(autocorr_range) or autocorr_range <= 0:
            raise ValueError("The estimated variogram range is not valid. "
                             "Check the variogram model and input coordinates.")
        
        # Use the estimated autocorrelation range as the
        # initial spatial block size.
        block_size = float(autocorr_range)
        self.spatial_block_size = block_size
        
        # Assign pixels to spatial blocks
        # Calculate the spatial block index for each pixel.
        # Coordinates must be in a projected coordinate system with units of metres.
        block_x = np.floor((coords[:, 0] - coords[:, 0].min()) / block_size).astype(int)
        block_y = np.floor((coords[:, 1] - coords[:, 1].min()) / block_size).astype(int)

        
        # Each unique combination of block_x and block_y represents one spatial block.
        block_indices = np.column_stack((block_x, block_y))
        _, groups = np.unique(block_indices,
                              axis=0,
                              return_inverse=True)
    
        n_blocks = len(np.unique(groups))                                    
        
        # Five-fold spatial CV requires at least five blocks.
        n_splits = 5
        if n_blocks < n_splits:
            raise ValueError(f"Only {n_blocks} spatial blocks were created. "
                             f"At least {n_splits} are required for "
                            "five-fold spatial cross-validation. "
                            "Consider testing a smaller block size.")
            
        # Initialize spatial cross-validation
        spatial_cv = GroupKFold(n_splits=n_splits)
        
        # Arrays to store the out-of-fold predictions and residuals.
        # Every valid observation will receive one prediction
        # from a model that did not use that observation for training.
        y_pred_oof = np.full(len(y), np.nan)
        deviance_residuals_oof = np.full(len(y), np.nan)
        fold_id = np.full(len(y), -1, dtype=int)
               
        # Perform five-fold spatial block cross-validation
        fold_metrics = []
        fold_moran_results = []
        for fold, (train_idx, test_idx) in enumerate(
                spatial_cv.split(X, y, groups=groups), start=1):

            # Split the observations according to spatial blocks.
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            # Build a fresh GAM structure for this fold.
            terms_fold, spline_type = self.BuildGAMTerms(Xs_list)   # terms, 

            # Fit the GAM using only the training observations.
            gam_fold = LinearGAM(terms_fold).gridsearch(X_train,
                                                        y_train,
                                                        lam=self.lam_grid)
            
            # Predict the observations in the test fold.
            y_pred = gam_fold.predict(X_test)
            
            # Calculate ordinary prediction residuals
            residuals = y_test - y_pred
            
            # Calculate deviance residuals for the test observations.
            dev_res = gam_fold.deviance_residuals(X_test,
                                                  y_test)
            
            # Store predictions and residuals at their original
            # positions in the complete arrays.
            y_pred_oof[test_idx] = y_pred
            deviance_residuals_oof[test_idx] = dev_res
            fold_id[test_idx] = fold
            
            # Calculate performance metrics for this fold.
            fold_mae = mean_absolute_error(y_test, y_pred)
            fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            fold_r2 = r2_score(y_test, y_pred)
            rmse_mae = fold_rmse / fold_mae
            y_range = np.nanmax(y_test) - np.nanmin(y_test)
            y_mean = np.nanmean(y_test)
            mae_yRange = fold_mae / y_range
            rmse_yRange = fold_rmse / y_range
            pseudo_r2 = gam_fold.statistics_["pseudo_r2"]["explained_deviance"] 
            
            fold_metrics.append({"Fold": fold,
                                 "N_train": len(train_idx),   
                                 "N_test": len(test_idx),
                                 "MAE": fold_mae,
                                 "RMSE": fold_rmse,
                                 "RMSE/MAE": rmse_mae,
                                 "MAE/Y Range": mae_yRange,
                                 "RMSE/Y Range": rmse_yRange,
                                 "R2": fold_r2,
                                 "Mean Y": y_mean})
            
            # Calculate residual spatial autocorrelation
            moran_results = self.CalculateResidualMoran(coords=coords[test_idx],
                                                        residuals=residuals,
                                                        fold=fold,
                                                        threshold=neighbor_distance,
                                                        permutations=4999)  # 499
            fold_moran_results.append(moran_results)
            
            fold_residuals = pd.DataFrame({"Fold": fold,
                                           "X": coords[test_idx, 0],
                                           "Y": coords[test_idx, 1],
                                           "Observed_Malleability": y_test,
                                           "Predicted_Malleability": y_pred,
                                           "Residual": residuals,
                                           "Block_ID": groups[test_idx]})
            
            self.PlotResidualMap(fold_residuals, 
                                 map_extent, 
                                 dirname=final_path, 
                                 fold_num=fold) 
            
            self.PartialDependencePlot(model_type="Linear GAM", 
                                       gam=gam_fold, 
                                       dep_var=dep_var, 
                                       column_names=list(X_df.columns),
                                       dirname=final_path, 
                                       fold_num=fold)
            
            self.DrawDevienceResidualsPlot(model_type="Linear GAM", 
                                           prediction=y_pred,
                                           devience_residuals=dev_res,
                                           dirname=final_path,
                                           fold_num=fold) 
            
            self.ModelSummary(model_type="Linear GAM", 
                              gam=gam_fold,   
                              dep_var=dep_var, 
                              indep_vars=list(X_df.columns),
                              splines=spline_type,
                              mae=fold_mae,
                              rmse=fold_rmse,
                              pseudo_r2=pseudo_r2,
                              r2=fold_r2,
                              y_range=y_range,
                              y_mean=y_mean,
                              dirname=final_path,
                              fold_num=fold)
            
        # Store the fold-wise results in a DataFrame.
        fold_metrics_df = pd.DataFrame(fold_metrics,
                                       columns=(["Fold", 
                                                 "N_train", 
                                                 "N_test", 
                                                 "MAE", 
                                                 "RMSE", 
                                                 "RMSE/MAE",
                                                 "MAE/Y Range",
                                                 "RMSE/Y Range",
                                                 "R2",
                                                 "Mean Y"]))
        
        residual_moran_df = pd.DataFrame(fold_moran_results)
        
        # Calculate the overall spatial-CV performance
        # At this point, each valid observation has one out-of-fold prediction.
        if np.isnan(y_pred_oof).any():
            raise RuntimeError("Some observations did not receive an "
                               "out-of-fold prediction.")
            
        # Calculate pooled metrics using all out-of-fold predictions.
        mae = mean_absolute_error(y, y_pred_oof)
        rmse = np.sqrt(mean_squared_error(y, y_pred_oof))
        r2 = r2_score(y, y_pred_oof)
        rmse_mae = rmse / mae
        
        # Calculate the standard deviation, range, and mean of the observed dependent variable.
        y_std = np.std(y)
        y_range = np.max(y) - np.min(y)
        y_mean = np.nanmean(y)      
        mae_yRange = mae / y_range
        rmse_yRange = rmse / y_range
        
        # Add overall spatial-CV performance
        overall_metrics = {"Fold": "Overall",
                           "N_train": np.nan,  
                           "N_test": len(y),
                           "MAE": mae,
                           "RMSE": rmse,
                           "RMSE/MAE": rmse_mae,
                           "MAE/Y Range": mae_yRange,
                           "RMSE/Y Range": rmse_yRange,
                           "R2": r2, 
                           "Mean Y": y_mean}
                                    
        fold_metrics_df.loc[len(fold_metrics_df)] = overall_metrics
                
        filename = f"{self.area}_GAM_results_on_five_fold_spatial_block_CV.txt"
        outpath = os.path.join(final_path, filename)
        
        with open(outpath, "w") as f:
            f.write(f"{'-' * 44} Variogram {'-' * 44}\n")
            f.write("Variogram Configuration:\n")             
            f.write(f"    Variogram model: {variogram_results['model']}\n")
            f.write(f"    Effective range: {self.spatial_autocorrelation_range}\n")
            f.write("\n\n")
            
            f.write(f"{'-' * 32} Results of GAM on Spatial Block CV {'-' * 31}\n")
            f.write("Model Configuration:\n")
            f.write(f"    Model type             : Linear GAM\n")
            f.write(f"    Dependent variable     : {dep_var}\n") 
            f.write(f"    Independent variable(s): {xs}\n")
            f.write("\n\n")
            
            f.write("Spatial Block CV Configuration:\n")
            f.write(f"    Estimated autocorrelation range: {autocorr_range:.2f} m\n")
            f.write(f"    Spatial block size             : {block_size:.2f} m\n")
            f.write(f"    Number of spatial blocks       : {n_blocks}\n")
            f.write(f"    Number of folds                : {n_splits}\n")
            f.write("\n\n")
            
            f.write("Model Performance:\n")            
            f.write(fold_metrics_df.to_string(index=False,
                                              float_format=lambda x: f"{x:.4f}"))
            f.write("\n\n\n")
            
            f.write(f"{'-' * 36} Residual Autocorrelation {'-' * 37}\n")
            f.write("Method Name: Global Moran's I\n")
            f.write(f"Neighbor Distance (the distance between diagonally adjacent pixel centres): {neighbor_distance:.2f} m\n\n")
            f.write("Residual Spatial Autocorrelation:\n")
            f.write(residual_moran_df.to_string(index=False,
                                                float_format=lambda x: f"{x:.5f}"))
            
            f.write(f"\n\n\n{'-' * 99}\n")
            f.write("WORKFLOW\n")
            f.write(f"* Variogram analysis is performed to estimate the spatial autocorrelation range of observed {dep_var}.\n")
            f.write("* Spatial block construction is performed to determine the block size and assign pixels to spatial blocks.\n")
            f.write("* Five GAMs were trained on five folds and out-of-fold predictions were generated.\n")
            f.write("* Fold-wise predictive performance were done (MAE, RMSE, and R²).\n")
            f.write("* Fold-specific residual autocorrelation was analzed by Calculating Moran's I for each test fold.\n")
            f.write("* Overall residual autocorrelation was analyzed by calculating Moran's I, using all out-of-fold residuals.\n")
            
            f.write("\n\nNOTES\n")
            f.write("** 'Neighbor Distance' is calculated using the X and Y dimensions of grids.\n")
            f.write("** 'GroupKFold' is applied to divide the blocks into five folds.\n")
            f.write("** The word 'Island' is used for grids that haven't neighbors.\n")
            
            f.write("\n\nInterpretations\n")
            f.write("  Moran's I Interpretation:\n")
            f.write("    Positive Moran's I :  Similar residuals cluster spatially.\n")
            f.write("    Moran's I near zero:  Little detectable global spatial autocorrelation.\n")
            f.write("    Negative Moran's I :  Neighboring residuals tend to be dissimilar.\n\n")
            f.write("    Positive, significant Moran's I in all folds : Consistent unexplained spatial patterns.\n")
            f.write("    Non-significant Moran's I in all folds       : No detectable global residual autocorrelation.\n")
            f.write("    Significant in some folds only               : Spatial autocorrelation varies across folds.\n")
            f.write("    Negative, significant Moran's I              : Spatially alternating prediction errors.")
        
        return final_path
    
    def MakeSmmaryCSVs(self, txt_file):
        study_area = None
        model_name = None
        fold = None
        dep_var = None
        indep_vars = ""
        mae = None
        rmse = None
        rmse_mae = None
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
                if line.startswith("Fold"):
                    fold = line.split(":")[-1].strip()    
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
                             
        model_data = [study_area, model_name, fold, dep_var, indep_vars, mae, rmse, 
                      rmse_mae, mae_yrane, rmse_yrane, r_squ, psudo_r_squ, y_mean]
            
        return model_data
        
    def Run(self):
        df, dest_pxl_size_x, dest_pxl_size_y = self.MakeDataframe()

        df.loc[df["Burn Severity"] < 0, "Burn Severity"] = np.nan
        Xs = ["Aspect", "Slope", "Altitude", "Burn Severity"]
        
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
                """
                for X_var in range(len(Xs)):
                    Xs_list = Xs[0:X_var+1]
                    self.LinearGAM(cdf, Xs_list, dep_var, dest_pxl_size_x, dest_pxl_size_y)
                """
                final_path = self.LinearGAM(cdf, Xs, dep_var, dest_pxl_size_x, dest_pxl_size_y)
                            
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
                """
                for X_var in range(len(Xs)):
                    Xs_list = Xs[0:X_var+1]
                    self.LinearGAM(cdf, Xs_list, dep_var, dest_pxl_size_x, dest_pxl_size_y)
                """
                final_path = self.LinearGAM(cdf, Xs, dep_var, dest_pxl_size_x, dest_pxl_size_y)
        
        col_name = ["Study Area", "Model", "Fold", "Dependent Var.", "Independent Vars.",
                    "MAE", "RMSE", "RMSE/MAE", "MAE/Y-Range", "RMSE/Y-Range", 
                    "R-Squared", "Pseudo R-Squared", "Mean Predicted Y"]
                
        summary_df = pd.DataFrame(columns=col_name)        
        
        for root, dirs, files in os.walk(final_path):   # self.outdir
            for n in files:
                if n.endswith("gam-summary.txt"):
                    txt_file = os.path.join(root, n)
                    model_data = self.MakeSmmaryCSVs(txt_file)
                    summary_df.loc[len(summary_df)] = model_data
                    area = model_data[0]
                    
        xlsx_filename = f"stat_summary.xlsx"
        xlsx_path = os.path.join(final_path, xlsx_filename)           
        summary_df.to_excel(xlsx_path, index=False)
        
        print(f"Outputs saved to:\n{final_path}")

    

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
          output_dir=out_path)

ins.Run()
