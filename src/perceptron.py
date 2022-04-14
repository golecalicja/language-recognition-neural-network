import numpy as np


class Perceptron:
    def __init__(self, weights, language, theta=0):
        np.random.seed(0)
        self.weights = weights
        self.language = language
        self.theta = theta

    def calculate_activation(self, row):
        dot_product = 0
        for i in range(len(row) - 1):
            dot_product += self.weights[i] * row[i]
        activation = dot_product - self.theta
        return activation

    def predict_classification(self, row):
        activation = self.calculate_activation(row)
        return self.unit_step_function(activation)

    def unit_step_function(self, activation):
        return 1 if activation >= self.theta else 0
