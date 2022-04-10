import numpy as np


class Perceptron:
    def __init__(self, number_of_weights, language):
        self.weights = np.random.uniform(low=-1, high=1, size=(number_of_weights,))
        self.theta = 0
        self.language = language

    def predict_classification(self, row):
        dot_product = 0
        for i in range(len(row[0]) - 1):
            dot_product += self.weights[i] * row[0][i]
        activation = dot_product - self.theta
        return self.unit_step_function(activation)

    def unit_step_function(self, activation):
        return 1 if activation >= self.theta else 0
