from src.algorithm_evaluator import AlgorithmEvaluator
from src.data_combiner import combine_texts
from src.data_cleaner import DataCleaner
from src.layer import Layer
from src.user_input_predictor import UserInputPredictor
from src.weights_trainer import WeightsTrainer

alpha = 0.01
number_of_epochs = 1000

train_directory = '../data/train/'
test_directory = '../data/test/'


def get_weights_trainers(df_train):
    weights_trainers = []
    languages = df_train['Language'].unique()
    train = df_train.to_numpy()
    for language in languages:
        weights_trainer = WeightsTrainer(train, alpha, number_of_epochs, language)
        weights_trainers.append(weights_trainer)
    return weights_trainers


def get_perceptrons(weights_trainers):
    perceptrons = []
    for weights_trainer in weights_trainers:
        perceptron = weights_trainer.train_weights()
        perceptrons.append(perceptron)
    return perceptrons


def get_prepared_train():
    df_train = combine_texts(train_directory)
    data_cleaner = DataCleaner(df_train)
    train = data_cleaner.vectorized()
    return train


def get_prepared_test():
    df_test = combine_texts(test_directory)
    data_cleaner = DataCleaner(df_test)
    test = data_cleaner.vectorized()
    return test


def main():
    df_train = get_prepared_train()
    df_test = get_prepared_test()
    test = df_test.to_numpy()

    weights_trainers = get_weights_trainers(df_train)
    perceptrons = get_perceptrons(weights_trainers)
    layer = Layer(perceptrons)
    algorithm_evaluator = AlgorithmEvaluator(layer, test)
    algorithm_evaluator.evaluate_model()
    user_input_predictor = UserInputPredictor(layer)
    user_input_predictor.predict_user_input()


if __name__ == '__main__':
    main()
