# %%


# Imports
import numpy as np

# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
y = np.array(([0], [1], [1], [0]), dtype=float)


# Define useful functions

# Activation function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:
    def __init__(self, x, y, hidden_layer_sizes):
        self.input = x
        layer_units = ([len(x[-1])] + hidden_layer_sizes + [1])

        self.weights = [np.random.rand(n_fan_in_, n_fan_out_) for n_fan_in_, n_fan_out_ in
                        zip(layer_units[:-1], layer_units[1:])]
        # self.weights1= np.random.rand(self.input.shape[1],4) # considering we have 4 nodes in the hidden layer
        # self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layers = [sigmoid(np.dot(self.input, self.weights[0]))]
        for layer in range(1, len(self.input) - 1):
            self.layers.append(sigmoid(np.dot(self.layers[-1], self.weights[layer])))
        # self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        # self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layers[-1]

    def backprop(self):
        d_weights = [np.dot(self.layers[-1].T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))]
        for layer in range(len(self.weights) - 2, -1, -1):
            layer_der = sigmoid_derivative(self.layers[-1])
            weight_l = self.weights[layer + 1].T
            d_weights.insert(0, np.dot(self.input.T,
                                       np.dot(d_weights[layer + 1],
                                              weight_l) * layer_der))
            # d_weights.insert(0, np.dot(self.input.T,
            #                            np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
            #                                   self.weights[layer + 1].T) * sigmoid_derivative(self.layers[-1])))
        # d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        # d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))

        print('weight', )
        for layer, weight in enumerate(self.weights):
            self.weights[layer] += d_weights[layer]
        # self.weights1 += d_weights1
        # self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


NN = NeuralNetwork(X, y, [2, 1])
for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("\n")

    NN.train(X, y)
