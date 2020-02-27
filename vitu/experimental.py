import tensorflow as tf
import pandas as pd
import numpy as np
import math

def create_example_schema(df: pd.DataFrame):
    schema = {}
    for key, dtype in df.dtypes.items():
        if dtype in (np.int64, np.bool):
            schema[key] = tf.feature_column.numeric_column(key, dtype=tf.int64)
        elif dtype in (np.float32, np.float64):
            schema[key] = tf.feature_column.numeric_column(key, dtype=tf.float32)
        elif dtype == np.object_:
            vocab = df[key].dropna().unique().tolist()
            _column = tf.feature_column.categorical_column_with_vocabulary_list(
                key,
                df[key].dropna().unique().tolist(),
            )
            schema[key] = tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    key,
                    vocab
                ),
                math.ceil(len(vocab) ** (1/4))
            )
        else:
            raise ValueError(f'unexpected dtype: column={key}, dtype={dtype}')
    return schema