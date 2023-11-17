from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
import numpy as np 

class random_forest_classifier:
    def __init__(self):
        self.name = "random_forest"
        self.description = "Random Forest Algorithm"
        self.model_category = "classification"
        self.is_trained = False
        self.hyperparameters = {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        }

    def set_hyperparameter(self, hyperparameter, value):
        self.hyperparameters[hyperparameter] = value

    def show_hyperparameters(self):
        arr_hyperparameter = ["  Hyperparameter  "]
        arr_value = [" Value "]
        
        print(f"Hyperparameters for {self.name}:\n")
        for hyperparameter, value in self.hyperparameters.items():
            #print(f"{hyperparameter}: {value}")
            arr_hyperparameter.append(hyperparameter)
            arr_value.append(value)

        return arr_hyperparameter, arr_value
        

    def run(self, dataset):
        if dataset is None:
            print("Please set the 'dataset' option first.")
            return None
        
        print(f"Training model using dataset '{dataset}'")

        if dataset == "iris":
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(**self.hyperparameters)
            train_start_time = time.time()
            model.fit(X_train, y_train)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time

            self.is_trained = True
            test_start_time = time.time()
            y_pred = model.predict(X_test)
            test_end_time = time.time()
            test_time = test_end_time - test_start_time

            accuracy = accuracy_score(y_test, y_pred)
            proba = model.predict_proba(X_test)

            if len(np.unique(y)) <= 2:
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
            else:
                f1 = f1_score(y_test, y_pred, average="weighted")
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")

            return {"train time": round(train_time, 4),
                    "test time": round(test_time, 4),
                    "accuracy": round(accuracy, 4),
                    "f1": round(f1, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4)}
        
        else:
            return "Dataset not supported."
