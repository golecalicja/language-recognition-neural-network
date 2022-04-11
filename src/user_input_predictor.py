from string import ascii_lowercase

from src.data_cleaner import normalized


def get_user_input():
    text = input('Enter text: ')
    return text


class UserInputPredictor:
    def __init__(self, layer):
        self.layer = layer
        self.text = get_user_input()

    def predict_user_input(self):
        vectorized_text = self.vectorized()
        prediction = self.layer.predict_output(vectorized_text)
        print('Predicted language: ' + prediction)

    def vectorized(self):
        letter_vector = self.create_letter_vector()
        normalized_vector = normalized(letter_vector)
        self.text = normalized_vector
        return self.text

    def create_letter_vector(self):
        letter_vector = []
        for letter in ascii_lowercase:
            letter_vector.append(self.text.count(letter))
        return letter_vector
