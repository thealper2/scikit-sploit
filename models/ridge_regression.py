from sklearn.linear_model import Ridge

class ridge_regression:
    def __init__(self):
        self.name = "ridge_regression"
        self.description = "Ridge Regression Algorithm"
        self.model_category = "regression"
        self.is_trained = False
        self.hyperparameters = {
            "alpha": 1,
            "fit_intercept": 1,
            "copy_X": 1,
            "max_iter": None,
            "solver": "auto",
            "positive": 0,
        }

    def set_hyperparameter(self, hyperparameter, value):
        self.hyperparameters[hyperparameter] = value
    
    def get_model(self):
        model = Ridge(**self.hyperparameters)
        return model