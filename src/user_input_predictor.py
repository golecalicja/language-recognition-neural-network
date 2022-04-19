import pandas as pd

from src.data_cleaner import DataCleaner


def get_user_input():
    text = input('Enter text: ')
    return text


class UserInputPredictor:
    def __init__(self, layer):
        self.layer = layer
        self.text = get_user_input()

    def transform_to_dataframe(self):
        data = {'Text': [self.text]}
        df = pd.DataFrame(data)
        return df

    def predict_user_input(self):
        vectorized_text = self.vectorized()
        prediction = self.layer.predict_output(vectorized_text)
        print('Predicted language: ' + prediction)

    def vectorized(self):
        df = self.transform_to_dataframe()
        data_cleaner = DataCleaner(df)
        return data_cleaner.vectorized()
