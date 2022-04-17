import numpy as np
import math


class Perceptron:
    def __init__(self, weights, language, theta=0):
        np.random.seed(0)
        self.weights = weights
        self.language = language
        self.theta = theta

    def calculate_dot_product(self, row):
        dot_product = 0
        for i in range(len(row) - 1):
            dot_product += self.weights[i] * row[i]
        return dot_product

    def predict_classification(self, row):
        dot_product = self.calculate_dot_product(row)
        return self.unit_step_function(dot_product)

    def unit_step_function(self, dot_product):
        return 1 if dot_product >= self.theta else 0

    def sigmoid_function(self, dot_product):
        activation = dot_product - self.theta
        return 1 / (1 + math.exp(-activation))
