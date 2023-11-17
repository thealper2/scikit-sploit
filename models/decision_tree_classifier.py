from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class decision_tree_classifier:
    def __init__(self):
        self.name = "decision_Tree"
        self.description = "Decision Tree Algorithm"
        self.model_category = "classification"
        self.is_trained = False
        self.hyperparameters = {
            "n_estimators": 100,
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": None
        }

    def set_hyperparameter(self, hyperparameter, value):
        self.hyperparameters[hyperparameter] = value

    def show_hyperparameters(self):
        print(f"Hyperparameters for {self.name}:")
        print(f"Dataset: {self.dataset}")
        for hyperparameter, value in self.hyperparameters.items():
            print(f"{hyperparameter}: {value}")

    def run(self, dataset):
        if dataset is None:
            print("Please set the 'dataset' option first.")
            return None
        
        print(f"Training model using dataset '{dataset}'")

        if dataset == "iris":
            data = load_iris()
            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

            model = DecisionTreeClassifier(**self.hyperparameters)
            model.fit(X_train, y_train)
            self.is_trained = True
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            return f"Model accuracy: {accuracy}"
        
        else:
            return "Dataset not supported."
