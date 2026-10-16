"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                               <<< SOME NOTES >>>                              #
#                                                                               #
#>>> This script categorizes the study areas in groups based on the given       #
#    parameters in the input file (for example amplitude, malleability, and     #
#    elasticity).                                                               #
#                                                                               #
#>>> The number of clusters is set by the argument 'cluster_range'. The argument#
#    is a list and could hold a single number or two numbers to present a range.#
#    The number of clusters could be set directly by seting single number in    #
#    list (for eaxmple [4]). If the argument contains two numbers (a range),    #
#    each number will be tested as the number of clusters and the best cluster  #
#    number will be selected based on the Silhouette score. The cluster number  #
#    with the highest Silhouette score will be selected as the cluster number   #
#    for further analysis. The script draws also a dendrogram plot using the    #
#    best cluster number.                                                       #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2026
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram



class PCA_analysis:
    def __init__(self, input_file, output_path, groups_column_name, cluster_range):
        self.input = input_file
        self.out = output_path
        self.groups_col = groups_column_name
        self.cluster_range = cluster_range
                
    def readFile(self):
        df = pd.read_excel(input_file)
        #print(df.columns)
        variables = [col for col in df.columns if col != self.groups_col]
        area_names = list(df["Study_Area"].values)
        X = df[variables].copy()
        return df, variables, area_names, X
    
    def variablesCorrelationMatrix(self, variables):
        output_file = "correlation_matrix_between_variables.txt"
        output_txt = os.path.join(self.out, output_file)
        
        with open(output_txt, "w") as file:
            with redirect_stdout(file):
                print("Correlation Matrix:\n")
                print(variables.corr())
                
        pd.plotting.scatter_matrix(variables,
                                   figsize=(8, 8),
                                   diagonal="hist")

        fig_name = "correlation_matrix_between_variables.png"
        fig_path = os.path.join(self.out, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
                
    def findBestClusterNumber(self, X_scaled):
        scores_of_clusters = {}
        labels_of_clusters = {}
        for k in range(self.cluster_range[0], self.cluster_range[1]):
            cluster = AgglomerativeClustering(n_clusters=k,
                                            linkage="ward")
            cluster_labels = cluster.fit_predict(X_scaled)
            score = silhouette_score(X_scaled,
                                     cluster_labels)
            scores_of_clusters[f"{k} Clusters"] = f"Silhouette Score = {score:.5f}"
            labels_of_clusters[f"{k} Clusters"] = cluster_labels
        
        output_file = "cluster_number_selection.txt"
        output_txt = os.path.join(self.out, output_file)
        
        msg = "NOTE: A higher Silhouette score indicates better-defined clusters."
        with open(output_txt, "w") as file:
            with redirect_stdout(file):
                print("Silhouette scores for examined cluster numbers:\n")
                for key, value in scores_of_clusters.items():
                    print(key, ":", value)     
        
        highet_score = max(list(scores_of_clusters.values()))
        best_cluster = [key for key, val in scores_of_clusters.items() if val == highet_score]
        best_cluster_number = int(best_cluster[0][:-9])
        labels_of_best_clusters = labels_of_clusters[best_cluster[0]]        
        
        with open(output_txt, "a") as file:
            with redirect_stdout(file):
                print(f"\nThe chosen cluster number: {best_cluster_number}")
                print(f"\n{msg}")
        
        return best_cluster_number, labels_of_best_clusters
    
    def byGivenClusterNumber(self, X_scaled):
        cluster_number = self.cluster_range[0]
        clustering = AgglomerativeClustering(n_clusters=self.cluster_range[0],
                                             linkage="ward")
        labels_of_clusters = clustering.fit_predict(X_scaled)
        
        return cluster_number, labels_of_clusters
                
    def PCAplot(self, df, X_pca, area_names, explained_variance):
        for cluster in sorted(df["Cluster"].unique()):
            mask = df["Cluster"] == cluster
            
            plt.scatter(X_pca[mask, 0],
                        X_pca[mask, 1],
                        s=40,
                        label=f"Cluster {cluster}")   

            for i in np.where(mask)[0]:
                if area_names[i] == "AD03":
                    pos = (-15, -12)
                elif area_names[i] == "AD02":
                    pos = (-15, 5)
                else:
                    pos = (-15, 5)
                    
                plt.annotate(area_names.iloc[i] if hasattr(area_names, "iloc") else area_names[i],
                             (X_pca[i, 0], X_pca[i, 1]),
                             xytext=pos,
                             textcoords="offset points")
        
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        
        plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)", fontsize=13)   
        plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)", fontsize=13)  
        plt.title("PCA Scores", fontsize=13)
        plt.legend()
        plt.grid(alpha=0.2, linestyle="--", linewidth=0.7)
        
        fig_name = "PCA_plot.png"
        fig_path = os.path.join(self.out, fig_name)     
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()   
    
    def loadingPlot(self, pca, variables):
        loadings = pca.components_.T
        
        label_offsets = {"Elasticity": (0.00, 0.05),
                         "Amplitude": (0.00, 0.05),
                         "Malleability": (0.00, -0.05)}
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        for i, variable in enumerate(variables):
            x = loadings[i, 0]  
            y = loadings[i, 1]  
            
            ax.arrow(0, 0,
                     x, y,
                     head_width=0.03,
                     head_length=0.03,
                     length_includes_head=True)
            
            dx, dy = label_offsets[variable]
            ax.text(x + dx,                                    
                    y + dy,                                    
                    variable.replace("_", " "),
                    fontsize=15,
                    ha="center",
                    va="center")
        
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.tick_params(axis="both", labelsize=13)
        
        ax.set_xlabel(f"PC1",    
                      fontsize=15)
        ax.set_ylabel(f"PC2",     
                      fontsize=15)
        ax.set_title("PCA Loading", fontsize=16)
        ax.grid(alpha=0.2)
        
        fig_name = "loading_plot.png"
        fig_path = os.path.join(self.out, fig_name)     
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close() 
        
    def hierarchicalClusteringWardMethod(self, X_scaled, area_names, cluster_number):
        Z = linkage(X_scaled, method="ward")
        distance_threshold = Z[-cluster_number + 1, 2]
        
        plt.figure(figsize=(10, 6))
        dendrogram(Z,
                   labels=area_names,
                   leaf_rotation=45,
                   leaf_font_size=10,
                   color_threshold=distance_threshold)

        plt.axhline(y=distance_threshold,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"{cluster_number} clusters")
        
        plt.xlabel("Study Area", fontsize=16)
        plt.ylabel("Ward Distance", fontsize=16)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.title("Hierarchical Clustering of Study Areas", fontsize=18)
        
        fig_name = "hierarchical_clustering_of_study_areas_by_Ward_method.png"
        fig_path = os.path.join(self.out, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    def PCA(self, df, cluster_number, X_scaled, variables, area_names):
        pca = PCA(n_components=len(variables))
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_
        
        scaled_df = pd.DataFrame()
        scaled_df[variables] = X_scaled
        scaled_df["Cluster"] = df["Cluster"]
        scaled_df["Resilience_score"] = df[variables].mean(axis=1)
        cluster_ranking = (scaled_df.groupby("Cluster")["Resilience_score"].mean().sort_values())       
        
        txt = "Explained variance:\n"
        for i, variance in enumerate(explained_variance):
            txt += f"PC{i+1}: {variance:.3f} ({variance*100:.1f}%)\n"
            
        col = [f"PC{i+1}" for i in range(len(explained_variance))]
        loadings = pd.DataFrame(pca.components_.T,
                                columns=col,
                                index=variables)
                
        output_file = "PCA_summary.txt"
        output_txt = os.path.join(self.out, output_file)
        
        note = "NOTE: PCA loadings tells how strongly each variable contributes to each component."
        with open(output_txt, "w") as file:
            with redirect_stdout(file):
                print(f"{txt}\n")
                print("PCA loadings:")
                print(loadings)
                print(f"\nCluster Ranking:\n{cluster_ranking}\n")
                print(f"\n{note}")
        
        self.PCAplot(df, X_pca, area_names, explained_variance)
        self.loadingPlot(pca, variables)
        self.hierarchicalClusteringWardMethod(X_scaled, area_names, cluster_number)
        
    def run(self):
        df, variables, area_names, X = self.readFile()
        X["Amplitude"] = -X["Amplitude"]         # A higher amplitde means lower resilience
        X["Malleability"] = -X["Malleability"]   # A higher Malleability means lower resilience
        self.variablesCorrelationMatrix(variables=X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if len(self.cluster_range) == 2:
            cluster_number, labels_of_clusters = self.findBestClusterNumber(X_scaled)
        elif len(self.cluster_range) == 1:
            cluster_number, labels_of_clusters = self.byGivenClusterNumber(X_scaled)
        
        cluster_labels = labels_of_clusters + 1
        X["Cluster"] = cluster_labels
        self.PCA(X, cluster_number, X_scaled, variables, area_names)
        
        print(f"The outputs are saved at:\n{self.out}")
        
       
#===================================== RUN ====================================
input_file = "E:/Data/PCA_for_Resilience/Resilience.xlsx"
output_path = "E:/Data/PCA_for_Resilience"

ins = PCA_analysis(input_file=input_file, 
                   output_path=output_path,
                   groups_column_name="Study_Area",
                   cluster_range=[2, 6])  

ins.run()