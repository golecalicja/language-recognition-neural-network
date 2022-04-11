class Layer:
    def __init__(self, test, perceptrons):
        self.test = test
        self.perceptrons = perceptrons

    def predict_output(self, row):
        language_to_activation_value = {}
        for perceptron in self.perceptrons:
            language_to_activation_value[perceptron.language] = perceptron.calculate_activation(row)
        print(language_to_activation_value)
        return max(language_to_activation_value, key=language_to_activation_value.get)

