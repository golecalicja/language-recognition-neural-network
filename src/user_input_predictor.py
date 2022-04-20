import pandas as pd

from src.data_cleaner import DataCleaner


def get_user_input():
    text = input('Enter text: \n')
    return text


class UserInputPredictor:
    def __init__(self, layer):
        self.layer = layer
        self.text = get_user_input()

    def predict_user_input(self):
        vectorized_text = self.vectorized()
        vectorized_text = vectorized_text.to_numpy()
        for row in vectorized_text:
            prediction = self.layer.predict_output(row[0])
            print('\n Predicted language: ' + prediction)

    def vectorized(self):
        df = self.transform_to_dataframe()
        data_cleaner = DataCleaner(df)
        return data_cleaner.vectorized()

    def transform_to_dataframe(self):
        d = {'Text': [self.text]}
        df = pd.DataFrame(data=d)
        return df
