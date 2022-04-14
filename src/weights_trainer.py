import numpy as np

from src.perceptron import Perceptron


def normalize(perceptron):
    perceptron.weights = perceptron.weights / np.sqrt(np.sum(perceptron.weights ** 2))


class WeightsTrainer:
    def __init__(self, train, alpha, number_of_epochs, language):
        self.train = train
        self.alpha = alpha
        self.number_of_epochs = number_of_epochs
        self.language = language

    def initialize_weights(self):
        return np.random.uniform(low=-1, high=1, size=(len(self.train) - 1,))

    def train_weights(self):
        weights = self.initialize_weights()
        perceptron = Perceptron(weights, self.language)
        for epoch in range(self.number_of_epochs):
            for row in self.train:
                prediction = perceptron.predict_classification(row[0])
                actual = row[-1] == self.language
                error = actual - prediction
                perceptron.theta -= self.alpha * error
                for i in range(len(row[0]) - 1):
                    perceptron.weights[i] += self.alpha * error * row[0][i]
                normalize(perceptron)

        return perceptron
