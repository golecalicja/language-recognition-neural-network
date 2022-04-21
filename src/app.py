from src.algorithm_evaluator import AlgorithmEvaluator
from src.data_cleaner import DataCleaner
from src.data_combiner import combine_texts
from src.layer import Layer
from src.user_input_predictor import UserInputPredictor
from src.weights_trainer import WeightsTrainer


def get_perceptrons(weights_trainers):
    perceptrons = []
    for weights_trainer in weights_trainers:
        perceptron = weights_trainer.train_weights()
        perceptrons.append(perceptron)
    return perceptrons


class App:
    def __init__(self, alpha, number_of_epochs, train_directory, test_directory):
        self.alpha = alpha
        self.number_of_epochs = number_of_epochs
        self.train_directory = train_directory
        self.test_directory = test_directory
        self.df_train, self.test = self.prepare_data()
        self.layer = self.create_layer()

    def get_prepared_train(self):
        df_train = combine_texts(self.train_directory)
        data_cleaner = DataCleaner(df_train)
        train = data_cleaner.vectorized()
        return train

    def get_prepared_test(self):
        df_test = combine_texts(self.test_directory)
        data_cleaner = DataCleaner(df_test)
        test = data_cleaner.vectorized()
        return test

    def get_weights_trainers(self, df_train):
        weights_trainers = []
        languages = df_train['Language'].unique()
        train = df_train.to_numpy()
        for language in languages:
            weights_trainer = WeightsTrainer(train, self.alpha, self.number_of_epochs, language)
            weights_trainers.append(weights_trainer)
        return weights_trainers

    def prepare_data(self):
        df_train = self.get_prepared_train()
        df_test = self.get_prepared_test()
        test = df_test.to_numpy()
        return df_train, test

    def prepare_perceptrons(self):
        weights_trainers = self.get_weights_trainers(self.df_train)
        perceptrons = get_perceptrons(weights_trainers)
        return perceptrons

    def create_layer(self):
        perceptrons = self.prepare_perceptrons()
        return Layer(perceptrons)

    def evaluate_model(self):
        algorithm_evaluator = AlgorithmEvaluator(self.layer, self.test)
        algorithm_evaluator.evaluate_model()

    def predict_user_input(self):
        user_input_predictor = UserInputPredictor(self.layer)
        user_input_predictor.predict_user_input()
