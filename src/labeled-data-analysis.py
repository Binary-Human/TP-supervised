""" 
Feature correlation analysis with labels
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class LabeledDataVisualizer:
    def __init__(self, csv_path, label_path=None, label_colname=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.label_df = None
        if label_path and label_colname:
            self.label_df = pd.read_csv(label_path)
            self.label = self.label_df[label_colname]

            self.df_labeled = pd.concat([self.df.reset_index(drop=True), pd.Series(self.label).reset_index(drop=True)], axis=1)
            self.df_labeled[label_colname] = pd.to_numeric(self.df_labeled[label_colname], errors="coerce")

        else:
            self.label = None
            self.df_labeled = None

    def feature_label_stats(self):
        # TODO : keep non numeric processing given Pearson only works on numeric ?
        """Affiche corr (Pearson) avec label et boxplots"""

        if self.df_labeled is None:
            raise RuntimeError("Aucun label attaché.")

        label_col = self.df_labeled.columns[-1]

        numerical_cols = [c for c in self.df_labeled.columns if c != label_col]

        res = {}
        for col in numerical_cols:
            ser = pd.to_numeric(self.df_labeled[col], errors='coerce')
            corr = ser.corr(self.df_labeled[label_col])
            res[col] = corr
            print(f"{col}: corr with label = {corr:.4f}")

            # boxplot
            try:
                self.df_labeled.boxplot(column=col, by=label_col, figsize=(10,5))
                plt.title(f"{col} by {label_col}")
                plt.suptitle("")
                plt.show()
            except Exception as e:
                print(f"Boxplot skipped for {col} : {e}")

        return pd.Series(res).sort_values(key=abs, ascending=False)

viz = LabeledDataVisualizer("2-Dataset/alt_acsincome_ca_features_85.csv", label_path="2-Dataset/alt_acsincome_ca_labels_85.csv", label_colname="PINCP")
viz.feature_label_stats()
