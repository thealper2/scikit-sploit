from sklearn.tree import DecisionTreeClassifier

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

    def get_model(self):
        model = DecisionTreeClassifier(**self.hyperparameters)
        return model