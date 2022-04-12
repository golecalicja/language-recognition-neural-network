from string import ascii_lowercase

import numpy as np


def distributed(letter_vector):
    vector = np.array(letter_vector)
    distributed_vector = vector / np.sum(vector)
    return distributed_vector


class DataCleaner:
    def __init__(self, df):
        self.df = df

    def vectorized(self):
        for i, row in self.df.iterrows():
            letter_vector = self.create_letter_vector(i)
            distributed_vector = distributed(letter_vector)
            self.df.at[i, 'Text'] = distributed_vector
        return self.df

    def create_letter_vector(self, i):
        letter_vector = []
        for letter in ascii_lowercase:
            letter_vector.append(self.df['Text'][i].count(letter))
        return letter_vector
