import os
import numpy as np
import pandas as pd
from string import ascii_lowercase

alpha = 0.0001
number_of_epochs = 1000


def to_letter_vector(df):
    for i, row in df.iterrows():
        letter_vector = []
        for letter in ascii_lowercase:
            letter_vector.append(df['Text'][i].count(letter))
        normalized_vector = normalize(letter_vector)
        df.at[i, 'Text'] = normalized_vector
    print(df)


def normalize(letter_vector):
    vector = np.array(letter_vector)
    normalized_vector = vector / np.sqrt(np.sum(vector ** 2))
    return normalized_vector


def all_texts_to_csv(path):
    languages = []
    for item in os.listdir(path):
        if os.path.isdir(path + item):
            languages.append(item)

    list_of_text = []
    for language in languages:
        for file in os.listdir(os.path.join(path, language)):
            with open(os.path.join(path, language, file), encoding='utf8', mode='r') as f:
                text = f.read()
            list_of_text.append((text, language))
        df = pd.DataFrame(list_of_text, columns=['Text', 'Language'])
        df.to_csv(os.path.join(path, '../new_csv_file.csv'))


def main():
    all_texts_to_csv('../data/train/')
    df = pd.read_csv('../data/new_csv_file.csv', index_col=0)
    print(df)
    to_letter_vector(df)


if __name__ == '__main__':
    main()
