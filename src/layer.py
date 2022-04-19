class Layer:
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons

    def predict_output(self, row):
        language_to_activation_value = {}
        for perceptron in self.perceptrons:
            dot_product = perceptron.calculate_dot_product(row)
            language_to_activation_value[perceptron.language] = perceptron.sigmoid_function(dot_product)
        return max(language_to_activation_value, key=language_to_activation_value.get)
