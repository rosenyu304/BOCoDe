import numpy as np

def happycat(X):

    r''' https://github.com/P-N-Suganthan/CEC2014/blob/master/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf

    "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary
    multi-objective optimization [educational forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87" '''

    # range of x = [-100, 100]

    X = 0.05 * X
    fx = np.abs(np.sum(X**2, axis=1) - X.shape[1])**0.25 + \
               (0.5 * np.sum(X**2, axis=1) + np.sum(X, axis=1)) / X.shape[1] + 0.5
    gx = 0
    return gx, fx


def discus(X):

    r''' https://github.com/P-N-Suganthan/CEC2014/blob/master/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf

    "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary
    multi-objective optimization [educational forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87" '''

    # range of x = [-100, 100]

    fx = 1e6 * X[:, 0]**2 + np.sum(X[:, 1:]**2, axis=1)
    gx = 0
    return gx, fx


def hgbat(X):

    r''' https://github.com/P-N-Suganthan/CEC2014/blob/master/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf

    "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary
    multi-objective optimization [educational forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87" '''

    # range of x = [-100, 100]

    X = 0.05 * X - 1  # Scale and shift the input
    fx = np.sqrt(np.abs(np.sum(X**2, axis=1)**2 - np.sum(X, axis=1)**2)) + \
                (0.5 * np.sum(X**2, axis=1) + np.sum(X, axis=1)) / X.shape[1] + 0.5
    gx = 0
    return gx, fx


def schaffer(X):
    r''' https://github.com/P-N-Suganthan/CEC2014/blob/master/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf

    "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform for evolutionary
    multi-objective optimization [educational forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87" '''

    # range of x = [-100, 100]

    X = X**2
    fx = np.sum(0.5 + (np.sin(np.sqrt(X + np.roll(X, -1, axis=1)))**2 - 0.5) /
                      (1 + 0.001 * (X + np.roll(X, -1, axis=1)))**2, axis=1)
    gx = 0
    return gx, fx

def sharp_ridge(X):

    r''' https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=70 '''

    # range of x = [-5, 5]

    fx = X^2 + 100*np.sqrt(np.sum(X**2, axis = 1))
    gx = 0
    return gx, fx

def different_powers(X):

    r''' https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=60 '''

    # range of x = [-5, 5]

    nx = len(X)
    fx = 0.0
    z = np.abs(X)

    for i in range(nx):
        f += np.power(z[i], 2 + 4 * i / (nx - 1))

    fx = np.power(f, 0.5)
    gx = 0
    return gx, fx
