import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import accuracy_score
from decision_tree import DecisionTreePipeline

import lime
import lime.lime_tabular
import shap


def permutation_feature_importance(model, x, y, n=10, random_state=42):
    r = np.random.RandomState(random_state)
    baseline_score = accuracy_score(y, model.predict(x))
    importances = []

    for feature in x.columns:
        scores = []
        for _ in range(n):
            x_permuted = x.copy()
            x_permuted[feature] = r.permutation(x_permuted[feature].values)
            scores.append(baseline_score - accuracy_score(y, model.predict(x_permuted)))
        importances.append(np.mean(scores))

    importance_df = pd.DataFrame({"Feature": x.columns, "Importance": importances}
                                ).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(10,6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Decrease in accuracy after permutation")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.show()
    return importance_df

model_path = "AdaBoost_BestModel_XXXX.joblib"
model = joblib.load(model_path)

pipeline = DecisionTreePipeline(model)
pipeline.load_data(
    "2-Dataset/alt_acsincome_ca_features_85.csv",
    "2-Dataset/alt_acsincome_ca_labels_85.csv",
)
pipeline.split_data()
pipeline.scale_column("AGEP", scaler_type="standard")
pipeline.scale_column("WKHP", scaler_type="minmax")


importance_df = permutation_feature_importance(pipeline.model, pipeline.features_test, pipeline.labels_test, n=10)
