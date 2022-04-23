import numpy as np

from src.perceptron import Perceptron


def normalize(perceptron):
    perceptron.weights = perceptron.weights / np.sqrt(np.sum(perceptron.weights ** 2))


class WeightsTrainer:
    def __init__(self, train, learning_rate, number_of_epochs, language):
        self.train = train
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.language = language

    def initialize_weights(self):
        np.random.seed(0)
        return np.random.uniform(low=-1, high=1, size=(len(self.train) - 1,))

    def train_weights(self):
        weights = self.initialize_weights()
        perceptron = Perceptron(weights, self.language)
        for epoch in range(self.number_of_epochs):
            self.train_epoch(perceptron)
        return perceptron

    def train_epoch(self, perceptron):
        for row in self.train:
            self.train_row(perceptron, row)

    def train_row(self, perceptron, row):
        letter_vector = row[0]
        actual_language = row[-1]
        prediction = perceptron.predict_classification(letter_vector)
        error = self.calculate_error(prediction, actual_language)
        self.update_bias(error, perceptron)
        self.update_weights(error, perceptron, letter_vector)
        normalize(perceptron)

    def calculate_error(self, prediction, actual_language):
        actual = actual_language == self.language
        error = actual - prediction
        return error

    def update_weights(self, error, perceptron, letter_vector):
        for i in range(len(letter_vector) - 1):
            perceptron.weights[i] += self.learning_rate * error * letter_vector[i]

    def update_bias(self, error, perceptron):
        perceptron.bias -= self.learning_rate * error
