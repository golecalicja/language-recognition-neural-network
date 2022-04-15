import numpy as np
import pandas as pd
from statsmodels.compat.pandas import assert_frame_equal

from src.data_cleaner import distributed, DataCleaner


def test_distributed():
    # given
    letter_vector = [1, 2, 3, 4]
    # when
    distributed_vector = distributed(letter_vector)
    # then
    assert np.array_equal(distributed_vector, [0.1, 0.2, 0.3, 0.4])


def test_vectorized():
    # given
    df = pd.DataFrame({'Text': ['aaabb'], 'Language': ['English']})
    data_cleaner = DataCleaner(df)
    # when
    result = data_cleaner.vectorized()
    # then
    vectorized_df = pd.DataFrame({'Text': [[0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                                  'Language': ['English']})
    assert_frame_equal(result, vectorized_df)

