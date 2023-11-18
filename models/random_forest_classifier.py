from sklearn.ensemble import RandomForestClassifier

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

    def get_model(self):
        model = RandomForestClassifier(**self.hyperparameters)
        return model