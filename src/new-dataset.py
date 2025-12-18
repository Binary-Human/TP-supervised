# load models from joblib
# import data
# test the models on data
# assess performance (accuracy, precision, recall...)
# Use my other modules if needed

import joblib

from decision_tree import DecisionTreePipeline

model_path = "Random_Forest_BestModel_XXXX.joblib"
model = joblib.load(model_path)

pipeline = DecisionTreePipeline(model)

pipeline.train_time = 0 # to avoid bug

pipeline.load_data(
    "4-Complementary data/Complementary data/acsincome_co_allfeatures.csv", 
    "4-Complementary data/Complementary data/acsincome_co_label.csv",
)

pipeline.split_data()
pipeline.scale_column("AGEP", scaler_type="standard")
pipeline.scale_column("WKHP", scaler_type="minmax")

print("\n--- Train set evaluation ---\n")
pipeline.evaluate(pipeline.features_train, pipeline.labels_train)
print("\n--- Test set evaluation ---\n")
pipeline.evaluate(pipeline.features_test, pipeline.labels_test)
pipeline.cross_validation()