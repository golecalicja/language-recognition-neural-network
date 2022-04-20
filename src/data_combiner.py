import os
import pandas as pd

combined_filename = 'texts_combined.csv'


def combine_texts_to_csv(path):
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
        df.to_csv(os.path.join(path, combined_filename))
