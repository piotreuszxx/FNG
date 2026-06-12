import numpy as np

def step(x):
    return 1 if x >= 0 else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=20, activation_function=step):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_function = activation_function

        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            for i in range(len(X)):

                z = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation_function(z)

                error = y[i] - prediction

                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def predict(self, x):

        z = np.dot(x, self.weights) + self.bias

        result = self.activation_function(z)

        # dla sigmoid/relu zamienia się wynik na klasę 0/1
        if self.activation_function != step:
            return 1 if result >= 0.5 else 0

        return result

    def evaluate(self, X, y):

        predictions = [self.predict(x) for x in X]

        correct = sum(pred == target for pred, target in zip(predictions, y))

        return correct / len(y)

X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1])

X_or = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_or = np.array([0, 1, 1, 1])

X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([0, 1, 1, 0])

def test_gate(name, X, y, activation_function):

    print("\n" + "=" * 50)
    print(f"{name} | aktywacja: {activation_function.__name__}")
    print("=" * 50)

    p = Perceptron(
        learning_rate=0.1,
        epochs=20,
        activation_function=activation_function
    )

    p.fit(X, y)

    print("Wagi:", p.weights)
    print("Bias:", p.bias)

    print("Predykcje:")

    for x in X:
        print(f"{x} -> {p.predict(x)}")

    print("Accuracy:", p.evaluate(X, y))


def test_gate(name, X, y, activation_function):

    print("\n" + "=" * 50)
    print(f"{name} | aktywacja: {activation_function.__name__}")
    print("=" * 50)

    p = Perceptron(
        learning_rate=0.1,
        epochs=20,
        activation_function=activation_function
    )

    p.fit(X, y)

    print("Wagi:", p.weights)
    print("Bias:", p.bias)

    print("Predykcje:")

    for x in X:
        print(f"{x} -> {p.predict(x)}")

    print("Accuracy:", p.evaluate(X, y))

test_gate("AND", X_and, y_and, step)
test_gate("AND", X_and, y_and, sigmoid)
test_gate("AND", X_and, y_and, relu)

test_gate("OR", X_or, y_or, step)
test_gate("OR", X_or, y_or, sigmoid)
test_gate("OR", X_or, y_or, relu)

test_gate("XOR", X_xor, y_xor, step)
test_gate("XOR", X_xor, y_xor, sigmoid)
test_gate("XOR", X_xor, y_xor, relu)