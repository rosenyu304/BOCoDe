import inspect
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

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
        # Find and load file
        model_file = cpath / Path("tools/microreactor_power_model.onnx")
        self.raw_model = ort.InferenceSession(str(model_file))

    def eval(self, pert):
        pert2 = pert.copy()
        # Reshape to 3D input as expected by the model: (batch_size, time_steps, features)
        pertn = np.array(
            [
                pert2,
            ]
        ).reshape(1, -1)
        unorm = self.raw_model.run(["dense_77"], {"input": pertn})[0]
        # Return a scalar objective value (sum of normalized power distribution)
        return float(unorm.sum())


def qPowerModel(pert):
    """Wrapper for QPowerModel that initializes and runs"""
    a = QPowerModel()
    return a.eval(pert)


if __name__ == "__main__":
    thetas = np.zeros(8)
    thetas[[6, 7]] -= np.pi
    print(qPowerModel(thetas))
