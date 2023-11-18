from sklearn.linear_model import LinearRegression

class linear_regression:
    def __init__(self):
        self.name = "linear_regression"
        self.description = "Linear Regression Algorithm"
        self.model_category = "regression"
        self.is_trained = False
        self.hyperparameters = {
            "fit_intercept": 1,
            "copy_X": 1,
            "n_jobs": None,
            "positive": 0
        }

    def set_hyperparameter(self, hyperparameter, value):
        self.hyperparameters[hyperparameter] = value
    
    def get_model(self):
        model = LinearRegression(**self.hyperparameters)
        return model