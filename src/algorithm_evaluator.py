def get_correct_predictions(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct


class AlgorithmEvaluator:
    def __init__(self, layer, test):
        self.layer = layer
        self.test = test

    def evaluate_model(self):
        correct, accuracy = self.calculate_accuracy()
        print('Correct predictions: %d' % correct)
        print('Accuracy: {:.1%}'.format(accuracy))

    def calculate_accuracy(self):
        actual = self.test[:, -1]
        predicted = []
        for row in self.test:
            prediction = self.layer.predict_output(row[0])
            predicted.append(prediction)
        correct = get_correct_predictions(actual, predicted)
        accuracy = correct / len(actual)
        return correct, accuracy
