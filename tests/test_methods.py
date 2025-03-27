import pandas as pd
import numpy as np
import pytest

from views_tensor_utilities import objects, mappings, defaults


def get_cm_index():
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


def get_pgm_index():
    index_tuples = [
        (492, 49182),
        (492, 49183),
        (492, 49184),
        (492, 49898),
        (492, 49899),
        (493, 49182),
        (493, 49183),
        (493, 49184),
        (493, 49898),
        (493, 49899),
        (494, 49182),
        (494, 49183),
        (494, 49184),
        (494, 49898),
        (494, 49899),
    ]

    return pd.MultiIndex.from_tuples(index_tuples)


def get_cm_float_data():
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


def get_cm_string_data():

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


def get_pgm_float_data():
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
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5
    ]

    return float_data


def get_cm_df():
    df_all = pd.DataFrame()

    df_all.index = get_cm_index()

    df_all['col_1'] = np.array(get_cm_float_data(), dtype=np.float64)
    df_all['col_2'] = np.array(get_cm_float_data(), dtype=np.float64)

    return df_all


def get_pgm_df():
    df_all = pd.DataFrame()

    df_all.index = get_pgm_index()

    df_all['col_1'] = np.array(get_pgm_float_data(), dtype=np.float64)
    df_all['col_2'] = np.array(get_pgm_float_data(), dtype=np.float64)

    return df_all


def get_cm_df_dtypes():
    df_all = pd.DataFrame()

    df_all.index = get_cm_index()

    df_all['32'] = np.array(get_cm_float_data(), dtype=np.float32)
    df_all['64'] = np.array(get_cm_float_data(), dtype=np.float64)

    return df_all


def get_cm_df_string():
    df_all = pd.DataFrame()

    df_all.index = get_cm_index()

    df_all['col_1'] = np.array(get_cm_string_data(), dtype=np.dtype('str'))
    df_all['col_2'] = np.array(get_cm_string_data(), dtype=np.dtype('str'))

    return df_all


def test_time_units():
    tu = mappings.TimeUnits.from_pandas(get_cm_index())

    assert np.array_equal(tu.times, np.array([492, 493, 494]))

    assert tu.time_to_index[492] == 0
    assert tu.time_to_index[493] == 1

    assert tu.index_to_time[2] == 494

    tu = mappings.TimeUnits.from_pandas(get_cm_df())

    assert np.array_equal(tu.times, np.array([492, 493, 494]))

    assert tu.time_to_index[492] == 0
    assert tu.time_to_index[493] == 1

    assert tu.index_to_time[2] == 494

    tu = mappings.TimeUnits.from_pandas(get_pgm_index())

    assert tu.time_to_index[492] == 0
    assert tu.time_to_index[493] == 1

    assert tu.index_to_time[2] == 494


def test_space_units():
    su = mappings.SpaceUnits.from_pandas(get_cm_index())

    assert np.array_equal(su.spaces, np.array([67, 69, 70, 73]))

    assert su.space_to_index[69] == 1
    assert su.space_to_index[73] == 3

    assert su.index_to_space[2] == 70

    su = mappings.SpaceUnits.from_pandas(get_cm_df())

    assert np.array_equal(su.spaces, np.array([67, 69, 70, 73]))

    assert su.space_to_index[69] == 1
    assert su.space_to_index[73] == 3

    assert su.index_to_space[2] == 70

    su = mappings.SpaceUnits.from_pandas(get_pgm_index())

    assert np.array_equal(su.spaces, np.array([49182, 49183, 49184, 49898, 49899]))

    assert su.space_to_index[49183] == 1
    assert su.space_to_index[49898] == 3

    assert su.index_to_space[2] == 49184


def test_long_lat_units():
    llu = mappings.LonglatUnits.from_pandas(get_pgm_index())

    assert np.array_equal(llu.pgids, np.array([49182, 49183, 49184, 49898, 49899]))
    assert llu.power == 3
    assert llu.gridsize == 8
    assert llu.latrange == 1
    assert llu.longrange == 6

    assert np.array_equal(np.array(llu.pgid_to_longlat[49183]), np.array([6, 3]))
    assert np.array_equal(np.array(llu.pgid_to_longlat[49899]), np.array([2, 4]))

    assert llu.longlat_to_pgid[7, 3] == 49184


def test_time_space_indices():
    tsu = mappings.TimeSpaceIndices.from_pandas(get_cm_index())

    assert tsu.ntime == 3
    assert tsu.nspace == 4
    assert tsu.nrow == 10

    assert np.array_equal(tsu.time_indices, np.array([492, 493, 494]))
    assert np.array_equal(tsu.space_indices, np.array([67, 69, 70, 73]))
    assert tsu.index_tuples[0] == (492, 69)
    assert tsu.index_tuples[6] == (493, 73)

    tsu = mappings.TimeSpaceIndices.from_pandas(get_cm_df())

    assert tsu.ntime == 3
    assert tsu.nspace == 4
    assert tsu.nrow == 10

    assert np.array_equal(tsu.time_indices, np.array([492, 493, 494]))
    assert np.array_equal(tsu.space_indices, np.array([67, 69, 70, 73]))
    assert tsu.index_tuples[0] == (492, 69)
    assert tsu.index_tuples[6] == (493, 73)


def test_is_strideable():
    assert not mappings.is_strideable(get_cm_index())
    assert mappings.is_strideable(get_pgm_index())


def test_get_index():
    try:
        mappings.get_index(get_cm_df())
    except:
        assert True


def test_check_df_data_types():

    assert mappings.__check_df_data_types(get_cm_df()) == np.dtype('float64')
    assert mappings.__check_df_data_types(get_cm_df_string()) == np.dtype('object')


def test_check_cast_to_dtypes():

    try:
        mappings.__check_cast_to_dtypes(np.dtype('float16'))
    except:
        assert True

    mappings.__check_cast_to_dtypes(np.dtype('float64'))
    mappings.__check_cast_to_dtypes(np.dtype('float32'))
    mappings.__check_cast_to_dtypes(np.dtype('int64'))
    mappings.__check_cast_to_dtypes(np.dtype('int32'))
    mappings.__check_cast_to_dtypes(np.dtype('str'))


def test_get_dne():

    try:
        mappings.get_dne(get_cm_df_dtypes())
    except:
        assert True

    try:
        assert mappings.get_dne(get_cm_df()) == defaults.float_dne
    except:
        assert mappings.get_dne(get_cm_df()) is defaults.float_dne

    assert mappings.get_dne(get_cm_df_string()) == '-----'


def test_get_tensor_dne():
    print('NOT IMPLEMENTED')


def test_get_missing():

    try:
        mappings.get_missing(get_cm_df_dtypes())
    except:
        assert True

    try:
        assert mappings.get_missing(get_cm_df()) == defaults.float_missing
    except:
        assert mappings.get_missing(get_cm_df()) is defaults.float_missing

    assert mappings.get_missing(get_cm_df_string()) == defaults.string_missing


def test_get_dtype():

    try:
        dtype = mappings.__get_dtype(get_cm_df_dtypes())
    except:
        assert True

    assert mappings.__get_dtype(get_cm_df()) == np.dtype('float64')
    assert mappings.__get_dtype(get_cm_df_string()) == np.dtype('object')


def test_df_to_numpy_time_space_strided():

    try:
        tensor_time_space = mappings.df_to_numpy_time_space_strided(get_cm_df())
    except:
        assert True

    tensor_time_space = mappings.df_to_numpy_time_space_strided(get_pgm_df())

    assert tensor_time_space.shape == (3, 5, 2)


def test_df_to_numpy_time_space_unstrided():

    tensor_time_space = mappings.df_to_numpy_time_space_unstrided(get_cm_df())

    assert tensor_time_space.shape == (3, 4, 2)

    assert tensor_time_space[0, 0, 0] == defaults.float_dne


def test_numpy_time_space_to_longlat():

    tensor_long_lat = mappings.numpy_time_space_to_longlat(mappings.df_to_numpy_time_space_unstrided(get_pgm_df()),
                                                           get_pgm_df().index)

    assert tensor_long_lat.shape == (8, 8, 3, 2)


def test_time_space_to_panel_unstrided():

    tensor_time_space = mappings.df_to_numpy_time_space_unstrided(get_cm_df())

    index = get_cm_index()

    columns = ['col_1', 'col_2']

    df = mappings.time_space_to_panel_unstrided(tensor_time_space, index, columns)

    assert df.equals(get_cm_df())


def test_time_space_to_panel_strided():

    tensor_time_space = mappings.df_to_numpy_time_space_unstrided(get_pgm_df())

    index = get_pgm_index()

    columns = ['col_1', 'col_2']

    df = mappings.time_space_to_panel_unstrided(tensor_time_space, index, columns)

    assert df.equals(get_pgm_df())


if __name__ == '__main__':
    test_time_units()
    test_space_units()
    test_long_lat_units()
    test_time_space_indices()
    test_is_strideable()
    test_get_index()
    test_check_df_data_types()
    test_check_cast_to_dtypes()
    test_get_dne()
    test_get_tensor_dne()
    test_get_missing()
    test_get_dtype()
    test_df_to_numpy_time_space_strided()
    test_df_to_numpy_time_space_unstrided()
    test_numpy_time_space_to_longlat()
    test_time_space_to_panel_unstrided()
    test_time_space_to_panel_strided()
