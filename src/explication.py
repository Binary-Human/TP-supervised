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


labels_pred = model.predict(pipeline.features_test)
y_true = pipeline.labels_test.values.ravel()
y_pred = labels_pred


# LIME

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=pipeline.features_train.values,
    feature_names=pipeline.features.columns.tolist(),
    class_names=["False", "True"],
    mode="classification"
)

sample_idx = np.random.choice(len(pipeline.features_test), 3, replace=False)

print("----- LIME explanations -----")
for idx in sample_idx:
    exp = explainer_lime.explain_instance(
        pipeline.features_test.iloc[idx].values,
        model.predict_proba,
        num_features=10
    )
    print(f"\nSample #{idx}:")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")

# SHAP

sample_idx = np.random.choice(len(pipeline.features_test), 5, replace=False)
X_sample = pipeline.features_test.iloc[sample_idx]

n_samples, n_features = X_sample.shape
shap_values_total = np.zeros((n_samples, n_features))

for weight, tree in zip(model.estimator_weights_, model.estimators_):
    explainer_tree = shap.TreeExplainer(tree)
    shap_values_tree = explainer_tree.shap_values(X_sample)

    if isinstance(shap_values_tree, list):
        contrib = shap_values_tree[1]
    elif shap_values_tree.ndim == 3:
        contrib = shap_values_tree[:, :, 1]
    elif shap_values_tree.ndim == 2:
        contrib = shap_values_tree[:, 1]
        contrib = np.tile(contrib, (n_samples, 1))

    else:
        raise ValueError(f"Format SHAP non supporté : {shap_values_tree.shape}")

    shap_values_total += weight * contrib

# Water plot
for i, idx in enumerate(sample_idx):
    print(f"\nSHAP Waterfall (AdaBoost approx) – sample #{idx}")
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_total[i],
            base_values=0,
            data=X_sample.iloc[i].values,
            feature_names=pipeline.features.columns.tolist()
        )
    )

print("\n----- SHAP Summary Plot Global -----")
shap.summary_plot(
    shap_values_total,
    X_sample,
    feature_names=pipeline.features.columns
)

# POur les sous-groupe
tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]
fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

groups = {
    "TP": np.where((y_true == 1) & (y_pred == 1))[0],
    "TN": np.where((y_true == 0) & (y_pred == 0))[0],
    "FP": np.where((y_true == 0) & (y_pred == 1))[0],
    "FN": np.where((y_true == 1) & (y_pred == 0))[0],
}

for name, idx_group in groups.items():
    if len(idx_group) == 0:
        continue

    X_group = pipeline.features_test.iloc[idx_group]
    shap_group = np.zeros((len(X_group), n_features))

    for weight, tree in zip(model.estimator_weights_, model.estimators_):
        explainer_tree = shap.TreeExplainer(tree)
        shap_tree = explainer_tree.shap_values(X_group)

        if isinstance(shap_tree, list):
            contrib = shap_tree[1]
        elif shap_tree.ndim == 3:
            contrib = shap_tree[:, :, 1]
        else:
            contrib = shap_tree[:, 1]
            contrib = np.tile(contrib, (len(X_group), 1))

        shap_group += weight * contrib

    print(f"\nSHAP Summary Plot – {name}")
    shap.summary_plot(shap_group, X_group, feature_names=pipeline.features.columns)
