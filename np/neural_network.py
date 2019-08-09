import numpy as np


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = NeuralNetwork.logistic
            self.activation_deriv = NeuralNetwork.logistic_derivative
        elif activation == 'tanh':
            self.activation = NeuralNetwork.tanh
            self.activation_deriv = NeuralNetwork.tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):  # add weights
            self.weights.append((2 * np.random.random(
                (layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            # self.weights.append((2 * np.random.random(
            #     (layers[i] + 1, layers[i + 1])) - 1) * 0.25)
        # b
        self.weights.append((2 * np.random.random(
            (layers[-2] + 1, layers[-1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)  # 转二维数组
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(
                    self.weights)):  # going forward network, for each layer
                # Computer the node value for each layer (O_i) using activation function
                ac = self.activation(np.dot(a[l], self.weights[l]))
                a.append(ac)
            error = y[i] - a[-1]  # Computer the error at the top layer
            # For output layer, Err calculation (delta is updated error)
            deltas = [error * self.activation_deriv(a[-1])]

            # Start backprobagation 后向算法
            for l in range(len(a) - 2, 0, -1):
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[l].T) *
                              self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    @staticmethod
    def tanh(x):  # 双曲函数
        return np.tanh(x)

    @staticmethod
    def tanh_deriv(x):  # 双曲函数导数
        return 1.0 - np.tanh(x) * np.tanh(x)

    @staticmethod
    def logistic(x):  # sigmoid 函数
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def logistic_derivative(x):  # sigmoid 函数导数
        return NeuralNetwork.logistic(x) * (1 - NeuralNetwork.logistic(x))


nn = NeuralNetwork([2, 20, 20, 1], 'tanh')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])
print(np.atleast_2d(X))
y = np.array([0, 1, 1, 0, 1])
nn.fit(X, y)
for i in [[0, 0], [0, 1], [1, 0], [2, 2]]:
    print(i, nn.predict(i))
