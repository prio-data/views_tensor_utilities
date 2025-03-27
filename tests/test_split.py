import pandas as pd
import numpy as np
import pytest

from views_tensor_utilities import objects, mappings, defaults


def get_index():
    index_tuples = [
        (492, 69),
        (492, 70),
        (492, 73),
        (493, 67),
        (493, 69),
        (493, 70),
        (493, 73),
        (494, 67),
        (494, 69),
        (494, 70),
    ]

    return pd.MultiIndex.from_tuples(index_tuples)


def get_other_space_index():
    index_tuples = [
        (492, 67),
        (492, 70),
        (492, 73),
        (493, 67),
        (493, 69),
        (493, 70),
        (493, 73),
        (494, 67),
        (494, 69),
        (494, 70),
    ]

    return pd.MultiIndex.from_tuples(index_tuples)


def get_other_time_index():
    index_tuples = [
        (493, 67),
        (493, 69),
        (493, 70),
        (493, 73),
        (494, 67),
        (494, 69),
        (494, 70),
        (495, 67),
        (495, 69),
        (495, 70),
    ]

    return pd.MultiIndex.from_tuples(index_tuples)


def get_float_data():
    float_data = [
        0.1,
        0.2,
        0.3,
        np.nan,
        0.5,
        0.6,
        0.7,
        np.nan,
        0.9,
        1.0
    ]

    return float_data


def get_int_data():

    int_data = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
    ]

    return int_data


def get_string_data():

    string_data = [
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine',
        'ten'
    ]

    return string_data


def build_test_df():

    df_all = pd.DataFrame()

    df_all.index = get_index()

    df_all['float32'] = np.array(get_float_data(), dtype=np.float32)
    df_all['float64'] = np.array(get_float_data(), dtype=np.float64)
    df_all['int32'] = np.array(get_int_data(), dtype=np.int32)
    df_all['int64'] = np.array(get_int_data(), dtype=np.int64)
    df_all['string'] = get_string_data()

    return df_all


def build_test_df_int_missing():

    df_all = pd.DataFrame()

    df_all.index = df_all.index = get_index()

    df_all['float32'] = np.array(get_float_data(), dtype=np.float32)
    df_all['float64'] = np.array(get_float_data(), dtype=np.float64)
    df_all['int32'] = np.array(get_int_data(), dtype=np.int32)
    df_all['int64'] = np.array(get_int_data(), dtype=np.int64)
    df_all['string'] = get_string_data()

    df_all.loc[df_all['int32'] == 5, 'int32'] = np.float32(np.nan)
    df_all['int32'] = df_all['int32'].astype(np.float32)
    df_all.loc[df_all['int64'] == 6, 'int64'] = np.float64(np.nan)

    return df_all


def build_test_df_dne_in_input():

    df_all = pd.DataFrame()

    df_all.index = df_all.index = get_index()

    df_all['float32'] = np.array(get_float_data(), dtype=np.float32)
    df_all['float64'] = np.array(get_float_data(), dtype=np.float64)
    df_all['int32'] = np.array(get_int_data(), dtype=np.int32)
    df_all['int64'] = np.array(get_int_data(), dtype=np.int64)
    df_all['string'] = get_string_data()

    df_all.loc[df_all['float32'] == 0.3, 'float32'] = defaults.float_dne
    df_all.loc[df_all['float64'] == 0.6, 'float64'] = defaults.float_dne

    return df_all


def build_test_df_other_space_index():

    df_all = pd.DataFrame()

    df_all.index = get_other_space_index()

    df_all['float32'] = np.array(get_float_data(), dtype=np.float32)
    df_all['float64'] = np.array(get_float_data(), dtype=np.float64)
    df_all['int32'] = np.array(get_int_data(), dtype=np.int32)
    df_all['int64'] = np.array(get_int_data(), dtype=np.int64)
    df_all['string'] = get_string_data()

    return df_all


def build_test_df_other_time_index():

    df_all = pd.DataFrame()

    df_all.index = get_other_time_index()

    df_all['float32'] = np.array(get_float_data(), dtype=np.float32)
    df_all['float64'] = np.array(get_float_data(), dtype=np.float64)
    df_all['int32'] = np.array(get_int_data(), dtype=np.int32)
    df_all['int64'] = np.array(get_int_data(), dtype=np.int64)
    df_all['string'] = get_string_data()

    return df_all


def build_test_tensors():

    float_data_1 = [
        [defaults.float_dne, 0.1, 0.2, 0.3],
        [0.5, np.nan, 0.6, 0.7],
        [0.8, np.nan, 0.9, defaults.float_dne]
    ]

    float_data_2 = [
        [defaults.float_dne, 1.1, 1.2, np.nan],
        [1.3, np.nan, 1.4, 1.5],
        [1.6, 1.7, np.nan, defaults.float_dne],
    ]

    int_data_1 = [
        [defaults.int_dne, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, defaults.int_dne],
    ]

    int_data_2 = [
        [defaults.int_dne, 11, 12, 13],
        [14, 15, 16, 17],
        [18, 19, 20, defaults.int_dne],
    ]

    int_data_3 = [
        [defaults.int_dne, 11, defaults.int_missing, 13],
        [14, 15, 16, 17],
        [18, 19, 20, defaults.int_dne],
    ]

    string_data_1 = [
        ['-----', 'one', 'two', 'three'],
        ['four', 'five', 'six', 'seven'],
        ['eight', 'nine', 'ten', '-----'],
    ]

    float_tensor = np.stack([np.array(float_data_1), np.array(float_data_2)], axis=2).astype(np.float64)

    float_columns = ['float_1', 'float_2']

    int_tensor = np.stack([np.array(int_data_1), np.array(int_data_2)], axis=2).astype(np.int64)

    int_columns = ['int_1', 'int_2']

    int_tensor_missing = np.stack([np.array(int_data_1), np.array(int_data_3)], axis=2).astype(np.int64)

    int_columns_missing = ['int_1', 'int_3']

    string_tensor = np.stack([np.array(string_data_1),], axis=2).astype(str)

    string_columns = ['str_1',]

    float_dtypes = [float_tensor.dtype, float_tensor.dtype]

    int_dtypes = [int_tensor.dtype, int_tensor.dtype]

    string_dtypes = [string_tensor.dtype,]

    index = get_index()

    float_views_tensor = objects.ViewsNumpy(float_tensor, columns=float_columns, dtypes=float_dtypes,
                                            dne=defaults.float_dne, missing=defaults.float_missing)

    float_views_tensor.index = index

    int_views_tensor = objects.ViewsNumpy(int_tensor, columns=int_columns, dtypes=int_dtypes,
                                          dne=defaults.int_dne, missing=defaults.int_missing)

    int_views_tensor.index = index

    int_views_tensor_missing = objects.ViewsNumpy(int_tensor_missing, columns=int_columns_missing, dtypes=int_dtypes,
                                                  dne=defaults.int_dne, missing=defaults.int_missing)

    int_views_tensor_missing.index = index

    string_views_tensor = objects.ViewsNumpy(string_tensor, columns=string_columns, dtypes=string_dtypes,
                                             dne=defaults.string_dne, missing=defaults.string_missing)

    string_views_tensor.index = index

    return [float_views_tensor, int_views_tensor, string_views_tensor, int_views_tensor_missing]


def test_split_df():

    df_all = build_test_df()

    views_df = objects.ViewsDataframe(df_all, 'float_string', 'to_64')

    assert len(views_df.split_dfs) == 2

    for dt in views_df.split_dfs[0].dtypes.values:
        assert dt is np.dtype('float64')

    views_df = objects.ViewsDataframe(df_all, 'float_string', 'to_32')

    for dt in views_df.split_dfs[0].dtypes.values:
        assert dt is np.dtype('float32')

    views_df = objects.ViewsDataframe(df_all, 'float_int_string', 'to_64')

    assert len(views_df.split_dfs) == 3

    for dt in views_df.split_dfs[0].dtypes.values:
        assert dt is np.dtype('float64')

    for dt in views_df.split_dfs[1].dtypes.values:
        assert dt is np.dtype('int64')

    views_df = objects.ViewsDataframe(df_all, 'maximal', 'to_64')

    assert len(views_df.split_dfs) == 5


def test_build_df_dne_in_input():

    df_all = build_test_df_dne_in_input()

    views_df = objects.ViewsDataframe(df_all, 'float_string', 'to_64')

    try:
        tensor_container = views_df.to_numpy_time_space()
    except:
        assert True


def test_tensor_build():

    df_all = build_test_df()

    views_df = objects.ViewsDataframe(df_all, 'float_string', 'to_64')

    tensor_container = views_df.to_numpy_time_space()

    assert len(tensor_container.ViewsTensors) == 2

    assert tensor_container.ViewsTensors[0].tensor.dtype is np.dtype('float64')

    assert tensor_container.ViewsTensors[0].tensor[0, 0, 0] == defaults.float_dne

    assert defaults.string_dne in tensor_container.ViewsTensors[1].tensor[0, 0, 0]

    assert np.isnan(defaults.float_missing)

    assert np.isnan(tensor_container.ViewsTensors[0].tensor[1, 0, 1])
    assert np.isnan(tensor_container.ViewsTensors[0].tensor[1, 0, 0])

    views_df = objects.ViewsDataframe(df_all, 'float_int_string', 'to_64')

    tensor_container = views_df.to_numpy_time_space()

    assert tensor_container.ViewsTensors[1].tensor[0, 0, 0] == defaults.int_dne
    assert defaults.string_dne in tensor_container.ViewsTensors[2].tensor[0, 0, 0]


def test_int_nan_handling():

    df_int_nans = build_test_df_int_missing()

    views_df = objects.ViewsDataframe(df_int_nans, 'float_string', 'to_64')

    tensor_container = views_df.to_numpy_time_space()

    assert tensor_container.ViewsTensors[0].tensor.dtype is np.dtype('float64')

    assert tensor_container.ViewsTensors[0].tensor[0, 0, 0] == defaults.float_dne

    assert np.isnan(tensor_container.ViewsTensors[0].tensor[1, 1, 2])
    assert np.isnan(tensor_container.ViewsTensors[0].tensor[1, 2, 3])

    views_df = objects.ViewsDataframe(df_int_nans, 'float_int_string', 'to_64')

    tensor_container = views_df.to_numpy_time_space()

    assert tensor_container.ViewsTensors[0].tensor.dtype is np.dtype('float64')

    assert len(tensor_container.ViewsTensors) == 2


def test_tensor_merger():

    tensors = build_test_tensors()

    assert tensors[0].tensor.dtype is np.dtype('float64')
    assert tensors[1].tensor.dtype is np.dtype('int64')

    try:
        merged_views_tensor = mappings.merge_views_tensors_to_views_tensor([tensors[0], tensors[1]])
    except:
        assert True

    merged_views_tensor = mappings.merge_views_tensors_to_views_tensor([tensors[0], tensors[1]],
                                                                       cast_to=np.float64, cast_to_dne=-np.inf,
                                                                       cast_to_missing=np.nan)

    assert merged_views_tensor.tensor[0, 0, 2] == defaults.float_dne
    assert merged_views_tensor.tensor[2, 3, 1] == defaults.float_dne

    tensors = build_test_tensors()

    merged_views_tensor = mappings.merge_views_tensors_to_views_tensor([tensors[0], tensors[1]],
                                                                       cast_to=np.int64, cast_to_dne=defaults.int_dne,
                                                                       cast_to_missing=defaults.int_missing)

    assert merged_views_tensor.tensor[0, 0, 0] == defaults.int_dne
    assert merged_views_tensor.tensor[1, 1, 0] == defaults.int_missing
    assert merged_views_tensor.tensor[2, 2, 1] == defaults.int_missing

    tensors = build_test_tensors()

    try:
        merged_views_tensor = mappings.merge_views_tensors_to_views_tensor([tensors[0], tensors[1],
                                                                            tensors[2]], cast_to=np.float64,
                                                                           cast_to_dne=-np.inf, cast_to_missing=np.nan)
    except:
        assert True

    tensors = build_test_tensors()

    merged_views_tensor = mappings.merge_views_tensors_to_views_tensor([tensors[0], tensors[3]],
                                                                       cast_to=np.float64,
                                                                       cast_to_dne=defaults.float_dne,
                                                                       cast_to_missing=defaults.float_missing)


def test_tensor_merger_bad_indexes():

    tensors_other = build_test_tensors()

    tensors_other[0].index = get_other_time_index()

    tensors = build_test_tensors()

    try:
        merged_views_tensor = mappings.merge_views_tensors_to_views_tensor([tensors[0],
                                                                            tensors_other[0]], cast_to=np.float64,
                                                                           cast_to_dne=-np.inf, cast_to_missing=np.nan)
    except:
        assert True

    tensors_other = build_test_tensors()

    tensors_other[0].index = get_other_space_index()

    try:
        merged_views_tensor = mappings.merge_views_tensors_to_views_tensor([tensors[0],
                                                                            tensors_other[0]], cast_to=np.float64,
                                                                           cast_to_dne=-np.inf, cast_to_missing=np.nan)
    except:
        assert True


def test_assemble_df_from_views_tensors():

    tensors = build_test_tensors()

    tensor_container = objects.ViewsTensorContainer.from_views_numpy_list(tensors)

    assert len(tensor_container.ViewsTensors) == 3

    tensors2 = build_test_tensors()

    tensors3 = tensors + tensors2

    tensor_container = objects.ViewsTensorContainer.from_views_numpy_list(tensors3)

    assert tensor_container.get_float_views_tensors()[0].tensor.shape[1] == 4
    assert len(tensor_container.get_float_views_tensors()[0].columns) == 4
    assert len(tensor_container.get_float_views_tensors()[0].dtypes) == 4
    assert tensor_container.get_int_views_tensors()[0].tensor.shape[1] == 4

    df = tensor_container.to_pandas()

    assert 'eight' in df['str_1'].values

#    assert len(df.columns) == 14


if __name__ == '__main__':
#    test_split_df()
#    test_tensor_build()
#    test_int_nan_handling()
    test_tensor_merger()
#    test_tensor_merger_bad_indexes()
#    test_assemble_df_from_views_tensors()
#    test_build_df_dne_in_input()
