import numpy as np
from pathlib import Path
import pandas as pd
from tensorflow.keras.models import load_model
import sys
import inspect

cpath = Path(inspect.getfile(sys.modules[__name__])).resolve().parent

def transform_features(x, f="cos"):
    if f == "cos":
        return np.cos(x)
    elif f == "sin":
        return np.sin(x)
    elif f == "tanh":
        return np.tanh(x)

class QPowerModel:
    """
    Use to evaluate quadrant power splits from control drum configurations.
    Set up as init, then separately use method call to minimize reading times.
    """
    def __init__(self):
        #Find and load file
        model_file = cpath / Path("tools/microreactor_power_model.keras")
        self.raw_model = load_model(model_file)

    def eval(self, pert):
        pert2 = pert.copy()
        # Reshape to 3D input as expected by the model: (batch_size, time_steps, features)
        pertn = np.array([pert2, ]).reshape(1, 1, -1)
        unorm = self.raw_model.predict(pertn).flatten()
        # Return a scalar objective value (sum of normalized power distribution)
        return float(unorm.sum())

def qPowerModel(pert):
    """Wrapper for QPowerModel that initializes and runs"""
    a = QPowerModel()
    return a.eval(pert)

if __name__ == "__main__":
    thetas = np.zeros(8)
    thetas[[6,7]] -= np.pi
    print(qPowerModel(thetas))
