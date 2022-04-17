from src.perceptron import Perceptron

perceptron = Perceptron([1, 2, 1, 1], 'English', 1)


def test_unit_step_function_should_return_1():
    # given
    dot_product = 1
    # when
    result = perceptron.unit_step_function(dot_product)
    # then
    assert result == 1


def test_unit_step_function_should_return_0():
    # given
    dot_product = 0
    # when
    result = perceptron.unit_step_function(dot_product)
    # then
    assert result == 0


def test_calculate_dot_product():
    # given
    row = [2, 2, 3, 4, 'English']
    # when
    result = perceptron.calculate_dot_product(row)
    # then
    assert result == 12


def test_predict_classification():
    # given
    row = [2, 2, 3, 4, 'English']
    # when
    result = perceptron.predict_classification(row)
    # then
    assert result == 1
