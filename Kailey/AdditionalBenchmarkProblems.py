import numpy as np

r''' "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary
multi-objective optimization [educational forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87" '''

def happycat(X):
    r''' https://github.com/P-N-Suganthan/CEC2014/blob/master/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf '''

    X = 0.05 * X  # Scale the input
    f = np.abs(np.sum(X**2, axis=1) - X.shape[1])**0.25 + \
               (0.5 * np.sum(X**2, axis=1) + np.sum(X, axis=1)) / X.shape[1] + 0.5
    return f


def discus(X):
    r''' https://github.com/P-N-Suganthan/CEC2014/blob/master/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf '''

    f = 1e6 * X[:, 0]**2 + np.sum(X[:, 1:]**2, axis=1)
    return f


def hgbat(X):
    r''' https://github.com/P-N-Suganthan/CEC2014/blob/master/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf '''

    X = 0.05 * X - 1  # Scale and shift the input
    f = np.sqrt(np.abs(np.sum(X**2, axis=1)**2 - np.sum(X, axis=1)**2)) + \
                (0.5 * np.sum(X**2, axis=1) + np.sum(X, axis=1)) / X.shape[1] + 0.5
    return f


def schaffer(X):
    r''' https://github.com/P-N-Suganthan/CEC2014/blob/master/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf '''

    X = X**2  # Square the input
    f = np.sum(0.5 + (np.sin(np.sqrt(X + np.roll(X, -1, axis=1)))**2 - 0.5) /
                      (1 + 0.001 * (X + np.roll(X, -1, axis=1)))**2, axis=1)
    return f


def rastrigin(X):
    r''' https://github.com/BIMK/PlatEMO/blob/da99f4cc5c9f5f6de96d8a2f1efdc65a49279e2f/PlatEMO/Problems/Single-objective%20optimization/CEC%202020/CEC2020_F9.m '''

    X = 0.0512 * X
    f = np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)
    return f
