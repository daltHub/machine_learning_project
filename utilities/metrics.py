"""
This file contains functions that help with evaluating performance of models
"""
from numpy.lib.function_base import average
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
import matplotlib
from numpy import mean

def test():
    """Test function"""
    print("Imported Correctly")

def calculate_mse(pred, true):

    losses = mean_squared_error(y_true=true, y_pred=pred)
    average_loss = mean(losses)
    print(f"average loss = {average_loss}")
    return average_loss

def dummy(X_features, y):
    x = DummyRegressor(strategy="mean")
    x.fit(X_features, y)
    return x.predict(X_features)