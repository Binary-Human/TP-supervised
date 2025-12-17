""" 
Effectuer une analyse exploratoire des données.

"""

# List all features
# Prelimiary reading of the data - on terminal

# Automate a script that for each feature provides:
# - Type of feature (numerical, categorical) -> extractable ?
# - Number of missing values
# - Basic statistics (mean, median...)
# - Visualizations (histograms, bar charts...)
# - Correlation analysis

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class DataVisualizer:
    def __init__(self, features_path):
        self.features_path = features_path
        self.df = pd.read_csv(features_path)
    
    def premilinary_analysis(self):
        print("Dataframe Info:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        print("\nBasic Statistics:")
        print(self.df.describe(include='all'))

    def visualize_one_parameter(self, parameter):
        if parameter not in self.df.columns:
            raise ValueError(f"Column '{parameter}' not found in dataframe.")

        data = self.df[[parameter]]

        plt.hist(data, bins=20, edgecolor='black', label=parameter)
        plt.xlabel(f'Distribution of {parameter}')
        plt.legend()
        plt.show()

    def visualize_all(self):
        for col in self.df.columns:
            self.visualize_one_parameter(col)

    def visualize_correlation(self):
        corr = self.df.corr()

        print("\n-------------------------------------\nTop correlations for each features:\n-------------------------------------\n")
        for col in corr.columns:
            top_corr = corr[col].abs().sort_values(ascending=False).head(4)
            # Exclude self-correlation
            top_corr = top_corr[top_corr.index != col]
            print(f"\n{col} correlations:")
            print(top_corr)

        plt.figure(figsize=(10, 8))
        plt.matshow(corr, fignum=1)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.colorbar()
        plt.title('Correlation Matrix', pad=20)
        plt.show()

    def top_global_correlations(self, top_n=10):
        corr = self.df.corr()
        
        print("\n-------------------------------------\nTop correlated features:\n-------------------------------------\n")

        corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))   # Keep values strictly above diagonal
                .stack()
                .dropna()
                .abs()
                .sort_values(ascending=False)
        )

        print(corr_pairs.head(top_n))

viz = DataVisualizer("2-Dataset/alt_acsincome_ca_features_85.csv")

viz.premilinary_analysis()
# viz.visualize_all()
# viz.visualize_one_parameter("AGEP")
# viz.visualize_correlation()
# viz.top_global_correlations(10)
