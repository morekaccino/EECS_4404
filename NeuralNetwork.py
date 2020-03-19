#!/usr/bin/env python
# coding: utf-8

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from copy import deepcopy


def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
    layer_units = ([len(X_train[-1])] + hidden_layer_sizes + [len(y_train[-1])])
    W = [np.empty((n_fan_in_ + 1, n_fan_out_)) for n_fan_in_,
                                                   n_fan_out_ in zip(layer_units[:-1],
                                                                     layer_units[1:])]



def forwardPropagation(x, weights):
    Xl = np.array(x)
    W = np.array(weights)
    S = []
    X = [x]
    for index, l in enumerate(W):
        wl = np.array(l)
        sl = np.transpose(wl).dot(Xl)
        Xl_before_activation = sl
        if index != len(W) - 1:
            activation_function = np.vectorize(activation)
            Xl = activation_function(Xl_before_activation)
            Xl = np.insert(Xl, 0, 1, axis=0)
        else:
            output_function = np.vectorize(outputf)
            Xl = output_function(Xl_before_activation)
        X.append(Xl)
        S.append(sl)
    np.delete
    return np.array(X), np.array(S)


def backPropagation(X, y_n, s, weights):
    weights_copy = deepcopy(weights)
    g = [None] * len(X)
    X = np.array(X)
    for layer, Xl in enumerate(reversed(X)):
        layer = len(X) - layer - 1
        if layer == len(X) - 1:
            delta = 2 * (Xl[0] - y_n) * derivativeActivation(s[-1][0])
            g[layer] = np.array([delta])
        elif layer > 0:
            derivatives = np.zeros([len(Xl) - 1, len(Xl) - 1])
            for i in range(len(Xl) - 1):
                derivatives[i][i] = derivativeActivation(Xl[i + 1])

            Wl = weights_copy[layer]
            Wl_t = np.array(Wl)
            g[layer] = ((Wl_t).dot((g[layer + 1]).T)[1:]).T.dot(derivatives)

    g = g[1:]

    updatedW = weights_copy
    for layer, Xl in enumerate(X[:-1]):
        updatedW[layer] = np.dot(np.array([Xl]).T, np.array([g[layer]]))

    return (updatedW)


def updateWeights(weights, g, alpha):
    nW = deepcopy(weights)
    for i in range(len(nW)):
        for j in range(len(nW[i])):
            nW[i][j] = nW[i][j] - (alpha * g[i][j])
    return nW


def activation(s):
    return 0 if s <= 0 else s


def derivativeActivation(s):
    return 0 if s <= 0 else 1


def outputf(s):
    return (1) / ((1) + np.exp(-s))


def derivativeOutput(s):
    return (outputf(s)) * (1 - outputf(s))


def errorf(x_L, y):
    if y == 1:
        return np.log(x_L)
    else:
        return -np.log(1 - x_L)


def errorPerSample(X, yn):
    return errorf(X[-1][-1], yn)


def derivativeError(x_L, y):
    if y == 1:
        # derivative of np.log(x_L)
        return 1 / (x_L)
    else:
        # derivative of -np.log(1 - x_L)
        return 1 / (1 - x_L)


def pred(x_n, weights):
    pass


def confMatrix(X_train, y_train, w):
    pass


def plotErr(e, epochs):
    pass


def test_SciKit(X_train, X_test, Y_train, Y_test):
    nn = MLPClassifier(hidden_layer_sizes=(300, 100), random_state=1, alpha=10 ** (-5))
    nn.fit(X_train, Y_train)
    pred = nn.predict(X_test)
    cm = confusion_matrix(Y_test, pred)
    return cm


def test():
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size=0.2)

    for i in range(80):
        if y_train[i] == 1:
            y_train[i] = -1
        else:
            y_train[i] = 1
    for j in range(20):
        if y_test[j] == 1:
            y_test[j] = -1
        else:
            y_test[j] = 1

    # err, w = fit_NeuralNetwork(X_train, y_train, 1e-2, [30, 10], 100)

    # plotErr(err, 100)

    # cM = confMatrix(X_test, y_test, w)

    sciKit = test_SciKit(X_train, X_test, y_train, y_test)

    # print("Confusion Matrix is from Part 1a is: ", cM)
    print("Confusion Matrix from Part 1b is:\n", sciKit)


test()
