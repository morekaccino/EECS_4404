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


# def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
#     layer_units = ([len(X_train[-1])] + hidden_layer_sizes + [1])
#     W = [np.random.rand(n_fan_in_ + 1, n_fan_out_) for n_fan_in_, n_fan_out_ in
#          zip(layer_units[:-1], layer_units[1:])]
#     W = np.true_divide(W, 10)
#     X_train = np.insert(X_train, 0, 1, axis=1)
#     # X_train = np.true_divide(X_train, 10 * len(X_train))
#     error_list = []
#     error_dic = {}
#     all_W = []
#     for _ in range(epochs):
#         error_over_epoch = 0
#         # np.random.shuffle(X_train)
#         for N, x_n in enumerate(X_train):
#             X_n, S_n = forwardPropagation(x_n, W)
#             g_n = backPropagation(X_n, y_train[N], S_n, W)
#             W = updateWeights(W, g_n, alpha)
#             all_W.append(W)
#             error_over_epoch += 1 if pred(x_n, W) != y_train[N] else 0
#         error_list.append(error_over_epoch / len(X_train))
#         error_dic[error_over_epoch] = W
#     # print(error_list)
#     print(np.subtract(all_W[-1], all_W[0]))
#     return error_list, W

def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
    layer_units = ([len(X_train[-1])] + hidden_layer_sizes + [1])
    w = [np.ones((n_fan_in_ + 1, n_fan_out_)) for n_fan_in_, n_fan_out_ in
         zip(layer_units[:-1], layer_units[1:])]
    w = np.true_divide(w, 10)
    X_train = np.insert(X_train, 0, 1, axis=1)

    for _ in range(epochs * 5):
        for data in zip(X_train, y_train):
            x, y = data
            if y == -1: y = 0
            print('x:', x, '\ty:', y)
            X, S = forwardPropagation(x, w)
            g, err = backPropagation(X, y, S, w)
            w = updateWeights(w, err, -alpha)
    print('done')
    print(w)
    return 0, w

# works for XOR
# def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
#     layer_units = ([len(X_train[-1])] + hidden_layer_sizes + [1])
#     w = [np.random.rand(n_fan_in_ + 1, n_fan_out_) for n_fan_in_, n_fan_out_ in
#          zip(layer_units[:-1], layer_units[1:])]
#
#     X_train = np.insert(X_train, 0, 1, axis=1)
#
#     for _ in range(epochs * 10):
#         for data in zip(X_train, y_train):
#             x, y = data
#             if y == -1: y = 0
#             print('x:', x, '\ty:', y)
#             X, S = forwardPropagation(x, w)
#             g, err = backPropagation(X, y, S, w)
#             w = updateWeights(w, err, -alpha)
#     print('done')
#     return w


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
    return np.array(X), np.array(S)


from copy import deepcopy


def backPropagation(X, y_n, s, weights):
    w = deepcopy(weights)
    g = [None] * len(X)
    X = np.array(X)
    for layer, Xl in enumerate(reversed(X)):
        layer = len(X) - layer - 1
        if layer == len(X) - 1:
            delta = 1 * (y_n - Xl[0]) * derivativeOutput(s[-1][0])
            g[layer] = np.array([delta])
        elif layer > 0:
            deltas = []
            for d in range(len(s[layer - 1])):
                derivative = derivativeActivation(s[layer - 1][d])
                sum = 0
                for k, delta in enumerate(g[layer + 1]):
                    sum += (delta * w[layer][d + 1][k])
                deltas.append(sum * derivative)
            g[layer] = np.array(deltas)

    g = g[1:]

    to_update_W = w
    for layer, Xl in enumerate(X[:-1]):
        to_update_W[layer] = np.dot(np.array([Xl]).T, np.array([g[layer]]))
    return g, to_update_W


def updateWeights(weights, err, alpha):
    return np.subtract(np.array(weights), np.multiply(np.array(err), alpha))
    # return np.add(weights, np.multiply(np.array(err), alpha))
    # return np.add(weights, np.array(err))


def activation(s):
    return 0 if s <= 0 else s
    # return 1 / (1 + np.exp(-s))


def derivativeActivation(s):
    return 0 if s <= 0 else 1
    # return activation(s) * (1 - activation(s))


def outputf(s):
    return (1) / (1 + np.exp(-s))
    # return np.tanh(s)


def derivativeOutput(s):
    return (outputf(s)) * (1 - outputf(s))
    # return 1 / np.cosh(s)


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
    x, s = forwardPropagation(x_n, weights)
    res = 1 if x[-1][-1] >= 0.5 else -1
    print('({}, {})'.format(res, x[-1][-1]))
    # res = 1 if x[-1][-1] >= 0 else -1
    return res


def confMatrix(X_train, y_train, w):
    # Add implementation here

    X_train = np.insert(X_train, 0, 1, axis=1)

    y_pred = []
    for x_n in X_train:
        y_pred.append(pred(x_n, w))

    # the confusion maxtrix that we will return
    # matrix = [[0, 0], [0, 0]]
    matrix = np.zeros((2, 2), np.int8)

    # Populating our matrix using the prediction data
    for index, y in enumerate(y_train):
        if y == -1 and y_pred[index] == -1:
            matrix[0][0] += 1
        elif y == -1 and y_pred[index] == 1:
            matrix[0][1] += 1
        elif y == 1 and y_pred[index] == -1:
            matrix[1][0] += 1
        else:
            matrix[1][1] += 1

    # returning the result
    return matrix
    # return confusion_matrix(y_train, y_pred)


def plotErr(e, epochs):
    x_axis = range(1, epochs + 1)
    plt.plot(e)
    plt.show()


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

    # X_train = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    # y_train = [-1, 1, 1, -1]
    # w = fit_NeuralNetwork(X_train, y_train, 1e1, [3], 100)
    # cM = confMatrix(X_train, y_train, w)

    err, w = fit_NeuralNetwork(X_train, y_train, 1e-2, [30, 10], 100)
    # plotErr(err, 100)
    cM = confMatrix(X_test, y_test, w)

    # sciKit = test_SciKit(X_train, X_test, y_train, y_test)

    print("Confusion Matrix is from Part 1a is:\n", cM)
    # print("Confusion Matrix from Part 1b is:\n", sciKit)


test()
