import tensorflow as tf
import pandas as pd
import numpy as np
import math
import jsondiff

class VituError(Exception):
    pass

def fillna(df: pd.DataFrame, default_float_value=0.0,
           default_string_value='', default_values={}) -> pd.DataFrame:
    _df = df.copy()
    for k, v in _df.dtypes.items():
        if _df[k].isnull().sum() == 0:
            continue
        if k in default_values:
            _default_value = default_values.get(k) 
        elif v == 'object':
            _default_value = default_string_value
        elif v == 'float':
            _default_value = default_float_value
        else:
            raise VituError(f'unexpected dtype: column={k}, dtype={v}')
        _df[k] = _df[k].fillna(_default_value)
    return _df

def diff_dtypes(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    return jsondiff.diff(df1.dtypes.to_dict(), df2.dtypes.to_dict(), syntax='symmetric')

def sync_dtypes(df1: pd.DataFrame, df2: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    _df1 = df1.copy()
    _df2 = df2.copy()
    diff = diff_dtypes(_df1, _df2)

    for k, dtypes in diff.items():
        if type(k) is not str:
            continue
        dtype1 = dtypes[0]
        dtype2 = dtypes[1]
        if 'object' in dtypes:
            if dtype1 != 'object':
                _df1[k] = _df1[k].astype('object')
            if dtype2 != 'object':
                _df2[k] = _df2[k].astype('object')
        elif 'float' in dtypes:
            if dtype1 != 'float':
                _df1[k] = _df1[k].astype('float')
            if dtype2 != 'float':
                _df2[k] = _df2[k].astype('float')
        else:
            raise VituError(f'unexpected dtypes:  column={k}, ' \
                            f'dtype1={dtype1}, dtype2={dtype2}')
    return _df1, _df2

def write_tfrecord(filename, df):
    if not tf.executing_eagerly():
        raise VituError('do not disable eager execution')

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _create_example_serialization(df):
        ser = {}
        for k, v in df.dtypes.items():
            if v == 'int':
                ser[k] = _int64_feature
            elif v ==  'float':
                ser[k] = _float_feature
            elif v ==  'object':
                ser[k] = _bytes_feature
            else:
                raise VituError(f'unexpected dtype: column={k}, dtype={v}')
        return ser

    example_serialization = _create_example_serialization(df)
    
    def serialize_example(features):
        feature = {k: example_serialization[k](v)
                   for k, v in features.items()}
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))
    
    def generator():
        for features in dataset:
            yield serialize_example(features)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)

def create_baseline_feature_columns(df: pd.DataFrame, ignore_keys=[]) -> list:
    def create_baseline_feature_column(key, dtype):
        if dtype == 'int':
            return tf.feature_column.numeric_column(key, dtype=tf.int64)
        elif dtype == 'float':
            return tf.feature_column.numeric_column(key, dtype=tf.float32)
        elif dtype == 'object':
            vocab = df[key].dropna().unique().tolist()
            _column = tf.feature_column.categorical_column_with_vocabulary_list(
                key,
                df[key].dropna().unique().tolist(),
            )
            return tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    key,
                    vocab
                ),
                math.ceil(len(vocab) ** (1/4))
            )
        else:
            raise VituError(f'unexpected dtype: column={key}, dtype={dtype}')

    return [create_baseline_feature_column(key, dtype)
            for key, dtype in df.dtypes.items()
            if key not in ignore_keys]


@DeprecationWarning
def read_csv(filepath_or_buffer, **kwargs):
    df = pd.read_csv(filepath_or_buffer, **kwargs)
    for k, v in df.dtypes.items():
        if v != np.object_:
            continue
        if df[k].isnull().sum() > 0:
            df[k] = df[k].fillna('')
    return df

@DeprecationWarning
def create_example_schema(df: pd.DataFrame):
    schema = {}
    for k in df.keys():
        _dtype = df[k].dtype
        if _dtype in (np.int64, np.bool):
            schema[k] = tf.int64
        elif _dtype in (np.float32, np.float64):
            schema[k] = tf.float32
        elif _dtype == np.float64:
            schema[k] = tf.float32
        elif _dtype == np.object_:
            schema[k] = tf.string
        else:
            raise VituError(f'unexpected dtype: column={k}, dtype={_dtype}')
    return schema


@DeprecationWarning
def unify_example_schema(l: dict, r: dict):
    for key, l_dtype in l.items():
        if key not in r:
            continue
        r_dtype = r.get(key)
        if l_dtype != r_dtype:
            if tf.string in (l_dtype, r_dtype):
                new_dtype = tf.string
            elif tf.float32 in (l_dtype, r_dtype):
                new_dtype = tf.float32
            elif tf.int64 in (l_dtype, r_dtype):
                new_dtype = tf.int64
            else:
                raise VituError(f'unexpected dtype: column={key},'
                                f'l_dtype={l_dtype}, r_dtype={r_dtype}')
            l[key] = new_dtype
            r[key] = new_dtype
    return l, r


@DeprecationWarning
def create_feature_columns(df: pd.DataFrame, ignore_keys=[]):
    def create_feature_column(key, dtype):
        if dtype in (np.int64, np.bool):
            return tf.feature_column.numeric_column(key, dtype=tf.int64)
        elif dtype in (np.float32, np.float64):
            return tf.feature_column.numeric_column(key, dtype=tf.float32)
        elif dtype == np.object_:
            vocab = df[key].dropna().unique().tolist()
            _column = tf.feature_column.categorical_column_with_vocabulary_list(
                key,
                df[key].dropna().unique().tolist(),
            )
            return tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    key,
                    vocab
                ),
                math.ceil(len(vocab) ** (1/4))
            )
        else:
            raise VituError(f'unexpected dtype: column={key}, dtype={dtype}')

    return [create_feature_column(key, dtype)
            for key, dtype in df.dtypes.items()
            if key not in ignore_keys]


@DeprecationWarning
def write_example_tfrecord(filename, dataset, schema):
    if not tf.executing_eagerly():
        raise VituError('do not disable eager execution')

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _create_example_serialization(schema):
        ser = {}
        for k, v in schema.items():
            if v ==  tf.int64:
                ser[k] = _int64_feature
            elif v ==  tf.float32:
                ser[k] = _float_feature
            elif v ==  tf.string:
                ser[k] = _bytes_feature
            else:
                raise VituError(f'unexpected dtype: column={k}, dtype={v}')
        return ser

    example_serialization = _create_example_serialization(schema)
    
    def serialize_example(features):
        feature = {k: example_serialization[k](v)
                   for k, v in features.items()}
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    def generator():
        for features in dataset:
            yield serialize_example(features)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)


@DeprecationWarning
def read_example_tfrecord(filename, schema):
    def _create_example_deserialization(schema):
        return {k: tf.io.FixedLenFeature([], v) for k, v in  schema.items()}

    example_deserialization = _create_example_deserialization(schema)

    def deserialize_example(features):
        return tf.io.parse_example(features, example_deserialization)

    def fillna(features):
        for k, v in schema.items():
            if v == tf.float32:
                features[k] = tf.where(
                    tf.math.is_nan(features[k]),
                    tf.zeros_like(features[k]),
                    features[k]
                )
        return features

    return tf.data.TFRecordDataset(filename).map(deserialize_example).map(fillna)

# pylint: disable=no-name-in-module,import-error
from tensorflow.python.feature_column.feature_column_v2 import NumericColumn, EmbeddingColumn


@DeprecationWarning
def is_numeric_column(column):
    return type(column) is NumericColumn


@DeprecationWarning
def is_embedding_column(column):
    return type(column) is EmbeddingColumn