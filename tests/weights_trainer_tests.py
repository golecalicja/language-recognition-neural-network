import numpy as np

from src.weights_trainer import WeightsTrainer

train = np.array([1, 1, 2, 1])
learning_rate = 0.1
number_of_epochs = 1000
language = 'English'
weights_trainer = WeightsTrainer(train, learning_rate, number_of_epochs, language)


def test_calculate_error_should_return_0():
    # given
    prediction = 1
    row = [1, 1, 2, 1, 'English']
    # when
    result = weights_trainer.calculate_error(prediction, row)
    # then
    assert result == 0


def test_calculate_error_should_return_1():
    # given
    prediction = 0
    row = [1, 1, 2, 1, 'English']
    # when
    result = weights_trainer.calculate_error(prediction, row)
    # then
    assert result == 1
