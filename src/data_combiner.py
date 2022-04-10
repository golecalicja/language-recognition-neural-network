import os
import pandas as pd


def combine_texts_to_csv(path):
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
        df.to_csv(os.path.join(path, 'texts_combined.csv'))


