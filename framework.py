import colorama
import time
import numpy as np

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
)

from colorama import Fore, Back, Style

colorama.init(autoreset=True)

def dynamic_table(*columns):
    max_length = [max(len(str(val)) for val in column) for column in zip(*columns)]
    
    for row in zip(*columns):
        table_row = "|"
        for i, val in enumerate(row):
            space_count = max_length[i] - len(str(val))
            table_row += f" {val}{' ' * space_count} |"

        print(table_row)

class ModelResult:
    def __init__(self, model_name, result):
        self.model_name = model_name
        self.result = result

class Framework:
    def __init__(self):
        self.dataset = None
        self.models = []
        self.classification_models = []
        self.regression_models = []
        self.clustering_models = []
        self.current_model = None
        self.results = {}
        self.prompt = f"{Fore.BLUE}sp {Fore.RESET}{Style.BRIGHT}> "

    def add_model(self, model):
        self.models.append(model)

    def set_dataset(self, dataset_name):
        self.dataset = dataset_name

    def list_models(self, category="All"):
        if category in ["All", "classification", "regression", "clustering"]:
            print(f"{category} models:")
            for index, model in enumerate(self.models, start=1):
                if category == "All":
                    print(f"{index}. {model.name}: {model.description}")
                elif model.model_category == category:
                    print(f"{index}. {model.name}: {model.description}")
        else:
            print("Invalid category.")

    def run(self):
        if self.current_model.model_category == "classification":
            result = self.run_classification()

        elif self.current_model.model_category == "regression":
            result = self.run_regression()

        return result

    def run_classification(self):
        if self.dataset == "iris":
            print(f"Training model using dataset '{self.dataset}'")

            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = self.current_model.get_model()

            train_start_time = time.time()
            model.fit(X_train, y_train)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time

            self.current_model.is_trained = True

            test_start_time = time.time()
            y_pred = model.predict(X_test)
            test_end_time = time.time()
            test_time = test_end_time - test_start_time

            accuracy = accuracy_score(y_test, y_pred)

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
    
    def run_regression(self):
        if self.dataset == "diabetes":
            print(f"Training model using dataset '{self.dataset}'")
            
            X, y = load_diabetes(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = self.current_model.get_model()
            
            train_start_time = time.time()
            model.fit(X_train, y_train)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time

            self.current_model.is_trained = True
            
            test_start_time = time.time()
            y_pred = model.predict(X_test)
            test_end_time = time.time()
            test_time = test_end_time - test_start_time

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            return {"train time": round(train_time, 4),
                    "test time": round(test_time, 4),
                    "r2": round(r2, 4),
                    "mae": round(mae, 4),
                    "mse": round(mse, 4),
                    "mape": round(mape, 4)}
        
        else:
            return "Dataset not supported."

    def show_results(self):
        if not self.results:
            print("No results available.")
            return
        
        print("Model Results:")
        arr_metric = ["  Metric  "]
        arr_metric_value = ["  Value  "]
        for metric, value in self.results.result.items():
            arr_metric.append(metric)
            arr_metric_value.append(value)

        dynamic_table(arr_metric, arr_metric_value)

    def show_help(self):
        print("Available commands")
        print("list models - List available models")
        print("use model X - Select a model by index")
        print("show hyperparameters - Show hyperparameters for selected model")
        print("set hyperparameter value - Set hyperparameter value for selected model")
        print("run - Execute the selected model")
        print("show results - Show results of executed models")
        print("help - Show available commands")

    def show_prompt(self):
        print(self.prompt, end="")

    def analyze_dataset(self):
        if self.current_model:
            if self.dataset:
                print("analyze dataset")

            else:
                print("Please, select a dataset.")
        else:
            print("Please select a model using the 'use model X' command.")

    def handle_command(self, command):
        if command == "help":
            self.show_help()

        elif command.startswith("list models"):
            parts = command.split(" ")
            if len(parts) == 3:
                category = parts[2]
                self.list_models(category=category)
            elif len(parts) == 2:
                self.list_models()

        elif command.startswith("use model"):
            parts = command.split(" ")
            if len(parts) == 3 and parts[2].isdigit():
                model_index = int(parts[2])
                if 0 < model_index <= len(self.models):
                    self.current_model = self.models[model_index - 1]
                    print(f"Using {self.current_model.name} model.")
                    self.prompt = f">/{self.current_model.name}/: "
                    self.prompt = f"{Fore.BLUE}sp {Fore.RESET}({Fore.RED}/{self.current_model.model_category}/{self.current_model.name}{Fore.RESET}) {Style.BRIGHT}>{Style.NORMAL} "
                else:
                    print("Invalid model index.")
            else:
                print("Invalid command format.")

        elif command == "analyze dataset":
            self.analyze_dataset()

        elif command == "run":
            if self.current_model:
                if self.dataset:
                    result = self.run()
                    model_result = ModelResult(self.current_model.name, result)
                    self.results = model_result
                else:
                    print("Choose a dataset first.")
            else:
                print("Please select a model using the 'use model X' command.")

        elif command == "show hyperparameters":
            if self.current_model:
                arr_hyperparameter = ["  Hyperparameter  "]
                arr_value = [" Value "]

                print(f"Hyperparameters for {self.current_model.name}\n")
                for hyperparameter, value in self.current_model.hyperparameters.items():
                    arr_hyperparameter.append(hyperparameter)
                    arr_value.append(value)

                dynamic_table(arr_hyperparameter, arr_value)
            else:
                print("Please select a model using the 'use model X' command.")

        elif command.startswith("set dataset"):
            if self.current_model:
                parts = command.split(" ")
                if len(parts) == 3:
                    self.set_dataset(parts[2])
                    print(f"Using dataset: {parts[2]}")
                else:
                    print("Error")
            else:
                print("Please select a model using the 'use model X' command.")

        elif command.startswith("set"):
            if self.current_model:
                parts = command.split(" ")
                if len(parts) == 3:
                    try:
                        self.current_model.set_hyperparameter(parts[1], int(parts[2]))
                    
                    except ValueError:
                        self.current_model.set_hyperparameter(parts[1], parts[2])

                else:
                    print("Error")

            else:
                print("Please select a model using the 'use model X' command.")

        elif command == "back":
            if self.current_model:
                self.current_model = None
                self.prompt = f"{Fore.BLUE}sp {Fore.RESET}{Style.BRIGHT}> "
                self.dataset = None

            else:
                print("No model selected already.")

        elif command == "show results":
            if self.current_model.is_trained == True:
                self.show_results()
            else:
                print("Please train model first")

        elif command == "exit":
            print("Exiting...")
            return True

        else:
            print("Invalid command. Use 'help' for list available commands.")