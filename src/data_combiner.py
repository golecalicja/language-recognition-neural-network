import os
import pandas as pd


def combine_texts(path):
    languages = []
    for item in os.listdir(path):
        if os.path.isdir(path + item):
            languages.append(item)

    texts = []
    for language in languages:
        for file in os.listdir(os.path.join(path, language)):
            with open(os.path.join(path, language, file), encoding='utf8', mode='r') as f:
                text = f.read()
            texts.append((text, language))
    df = pd.DataFrame(texts, columns=['Text', 'Language'])
    return df
