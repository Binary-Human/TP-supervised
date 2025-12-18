import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.utils import resample

import joblib

class DecisionTreePipeline:
    def __init__(self, model, model_type = GradientBoostingClassifier, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.model_type = model_type
        self.model = model
        self.train_time = None
        self.predict_time = None

    def load_data(self, features_path, labels_path):
        self.features = pd.read_csv(features_path)
        self.labels = pd.read_csv(labels_path)
        return self.features, self.labels
    
    def drop_featuress(self, columns): # helper functio
        self.features = self.features.drop(columns=columns)
        return self.features

    def split_data(self):
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(
            self.features,
            self.labels,
            test_size=self.test_size,
            shuffle=True,
            random_state=self.random_state,
        )
        return self.features_train, self.features_test, self.labels_train, self.labels_test

    def scale_column(self, column_name, scaler_type="standard"):

        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
        }

        if scaler_type not in scalers:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        scaler = scalers[scaler_type]

        # fit on train only
        self.features_train[[column_name]] = scaler.fit_transform(
            self.features_train[[column_name]]
        )

        self.features_test[[column_name]] = scaler.transform(
            self.features_test[[column_name]]
        )

    def train(self):
        start = time.time()
        self.model.fit(self.features_train, self.labels_train)
        self.train_time = time.time() - start
        return self.model
    
    def evaluate(self, x, y ):
        start = time.time()
        labels_pred = self.model.predict(x)
        self.predict_time = time.time() - start


        accuracy = accuracy_score(y, labels_pred)
        precision = precision_score(y, labels_pred)
        recall = recall_score(y, labels_pred)
        cm = confusion_matrix(y, labels_pred)

        print("\n----------- Model Evaluation -----------\n")
        print(f"Train time       : {self.train_time:.3f}s")
        print(f"Predict time     : {self.predict_time:.3f}s")
        print(f"Accuracy         : {accuracy:.4f}")
        print(f"Precision        : {precision:.4f}")
        print(f"Recall           : {recall:.4f}")
        print("\nConfusion Matrix:")
        print(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    def cross_validation(self, cv=5):
        
        labels_pred = cross_val_predict(self.model, self.features_test, self.labels_test, cv=cv)

        accuracy = cross_val_score(self.model, self.features_test, self.labels_test, cv=cv, scoring="accuracy")
        precision = cross_val_score(self.model, self.features_test, self.labels_test, cv=cv, scoring='precision')
        recall = cross_val_score(self.model, self.features_test, self.labels_test, cv=cv, scoring='recall')
        cm = confusion_matrix(self.labels_test, labels_pred)

        print("\n----------- Cross-Validation Evaluation -----------\n")
        print(f"CV folds         : {cv}")
        print(f"Mean Accuracy    : {accuracy.mean():.4f}")
        print(f"Mean Precision   : {precision.mean():.4f}")
        print(f"Mean Recall      : {recall.mean():.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(classification_report(self.labels_test, labels_pred))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    def grid_search_impl(self, param_grid):
        
        MAX_SAMPLES = 166316

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1
        )

        feature_train_resampled, labels_train_resampled = resample(
            self.features_train,
            self.labels_train,
            n_samples=MAX_SAMPLES,
            random_state=42,
            stratify=self.labels_train
        )
        
        start = time.time()
        grid_search.fit(feature_train_resampled, labels_train_resampled)
        exec_time = time.time() - start

        best_params = grid_search.best_params_

        print("Best hyperparameters:", grid_search.best_params_)
        print("Execution time:", exec_time)
        print("Best CV accuracy:", grid_search.best_score_)

        self.model = self.model_type( random_state=None, **best_params)

        self.train()

        print("\n--- Train set evaluation ---\n")
        pipeline.evaluate(self.features_train, self.labels_train)
        print("\n--- Test set evaluation ---\n")
        pipeline.evaluate(self.features_test, self.labels_test)

        joblib.dump(grid_search.best_estimator_,'BestModel_XXXX.joblib')

pipeline = DecisionTreePipeline(model=GradientBoostingClassifier())

pipeline.load_data(
    "2-Dataset/alt_acsincome_ca_features_85.csv",
    "2-Dataset/alt_acsincome_ca_labels_85.csv",
)

pipeline.split_data()
pipeline.scale_column("AGEP", scaler_type="standard")
pipeline.scale_column("WKHP", scaler_type="minmax")

""" 
pipeline.train()

print("\n--- Train set evaluation ---\n")
pipeline.evaluate(pipeline.features_train, pipeline.labels_train)
print("\n--- Test set evaluation ---\n")
pipeline.evaluate(pipeline.features_test, pipeline.labels_test)
pipeline.cross_validation()  """


param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

param_grid_adaB =  {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1],
}

param_grid_gBoost = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2, 1],
    "subsample": [0.5, 0.7, 1.0]
    
}

# pipeline.grid_search_impl(param_grid_gBoost)
