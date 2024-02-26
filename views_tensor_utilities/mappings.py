import numpy as np
import pandas as pd
from . import defaults


class TimeUnits():

    """
    TimeUnits

    Class which generates and holds a set of time indices and two dictionaries to quickly transform
    between them.

    A factory method to build instances from pandas dataframes or multindexes is included

    """

    def __init__(self, times, index_to_time, time_to_index):
        self.times = times
        self.index_to_time = index_to_time
        self.time_to_index = time_to_index

    @classmethod
    def from_pandas(cls, pandas_object):
        if isinstance(pandas_object, pd.DataFrame):
            index = pandas_object.index
        elif isinstance(pandas_object, pd.MultiIndex):
            index = pandas_object
        else:
            raise RuntimeError(f'Input is not a df or a df index')

        # get unique times

        times = np.array(list({idx[0] for idx in index.values}))
        times = list(np.sort(times))

        # make dicts to transform between times and the index of a time in the list

        time_to_index = {}
        index_to_time = {}
        for i, time in enumerate(times):
            time_to_index[time] = i
            index_to_time[i] = time

        time_units = cls(times=times, time_to_index=time_to_index, index_to_time=index_to_time)

        return time_units


class SpaceUnits():
    """
    SpaceUnits

    Class which generates and holds a set of space indices and two dictionaries to quickly transform
    between them.

    A factory method to build instances from pandas dataframes or multindexes is included

    """

    def __init__(self, spaces, index_to_space, space_to_index):
        self.spaces = spaces
        self.index_to_space = index_to_space
        self.space_to_index = space_to_index

    @classmethod
    def from_pandas(cls, pandas_object):
        if isinstance(pandas_object, pd.DataFrame):
            index = pandas_object.index
        elif isinstance(pandas_object, pd.MultiIndex):
            index = pandas_object
        else:
            raise RuntimeError(f'Input is not a df or a df index')

        spaces = np.array(list({idx[1] for idx in index.values}))

        spaces = np.sort(spaces)

        space_to_index = {}
        index_to_space = {}

        for i, space_id in enumerate(spaces):
            space_to_index[space_id] = i
            index_to_space[i] = spaces

        space_units = cls(spaces=spaces, space_to_index=space_to_index, index_to_space=index_to_space)

        return space_units


class LonglatUnits():
    """
    LonglatUnits

    Class which generates and holds a set of space indices, a set of matching longitude-latitude
    tuples and dictionaries to quickly transform between them.

    A factory method to build instances from pandas dataframes or multindexes is included

    """

    def __init__(self,
                 pgids,
                 pgid_to_longlat,
                 longlat_to_pgid,
                 pgid_to_index,
                 index_to_pgid,
                 longrange,
                 latrange,
                 gridsize,
                 power):

        self.pgids = pgids
        self.pgid_to_longlat = pgid_to_longlat
        self.longlat_to_pgid = longlat_to_pgid
        self.pgid_to_index = pgid_to_index
        self.index_to_pgid = index_to_pgid
        self.longrange = longrange
        self.latrange = latrange
        self.gridsize = gridsize
        self.power = power

    @classmethod
    def from_pandas(cls, pandas_object):
        """
        from_pandas

        Factory method which builds class instances from pandas dataframes or multiindexes

        """

        if isinstance(pandas_object, pd.DataFrame):
            index = pandas_object.index
        elif isinstance(pandas_object, pd.MultiIndex):
            index = pandas_object
        else:
            raise RuntimeError(f'Input is not a df or a df index')

        pgids = np.array(list({idx[1] for idx in index.values}))
        pgids = np.sort(pgids)

        # convert pgids to longitudes and latitudes

        longitudes = pgids % defaults.pg_stride
        latitudes = pgids // defaults.pg_stride

        latmin = np.min(latitudes)
        latmax = np.max(latitudes)
        longmin = np.min(longitudes)
        longmax = np.max(longitudes)

        latrange = latmax - latmin
        longrange = longmax - longmin

        # shift to a set of indices that starts at [0,0]

        latitudes -= latmin
        longitudes -= longmin

        # find smallest possible square grid with side 2^ncells which will fit the pgids

#        latmin = np.min(latitudes)
        latmax = np.max(latitudes)
#        longmin = np.min(longitudes)
        longmax = np.max(longitudes)

        maxsize = np.max((longrange, latrange))
        power = 1 + int(np.log2(maxsize))

        gridsize = 2 ** power

        # centre the pgids

        inudgelong = int((gridsize - longmax) / 2)
        inudgelat = int((gridsize - latmax) / 2)

        longitudes += inudgelong
        latitudes += inudgelat

        # make dicts to transform between pgids and (long,lat) coordinate

        pgid_to_longlat = {}
        longlat_to_pgid = {}

        pgid_to_index = {}
        index_to_pgid = {}

        for i, pgid in enumerate(pgids):
            pgid_to_longlat[pgid] = (longitudes[i], latitudes[i])
            longlat_to_pgid[(longitudes[i], latitudes[i])] = pgid
            pgid_to_index[pgid] = i
            index_to_pgid[i] = pgid

        longlat_units = cls(pgids=pgids,
                            pgid_to_longlat=pgid_to_longlat,
                            longlat_to_pgid=longlat_to_pgid,
                            pgid_to_index=pgid_to_index,
                            index_to_pgid=index_to_pgid,
                            longrange=longrange,
                            latrange=latrange,
                            gridsize=gridsize,
                            power=power)

        return longlat_units


class TimeSpaceIndices():
    """
    TimeSpaceUnits

    Class which generates and holds a set of time and space indices and a list of tuples,
    derived from a pandas df or multiindex.

    A factory method to build instances from pandas dataframes or multindexes is included

    """

    def __init__(self, time_indices, space_indices, index_tuples, ntime, nspace, nrow):
        self.time_indices = time_indices
        self.space_indices = space_indices
        self.index_tuples = index_tuples
        self.ntime = ntime
        self.nspace = nspace
        self.nrow = nrow

    @classmethod
    def from_pandas(cls, pandas_object):
        """
        from_pandas

        Factory method which builds class instances from pandas dataframes or multiindexes

        """

        if isinstance(pandas_object, pd.DataFrame):
            index = pandas_object.index
        elif isinstance(pandas_object, pd.MultiIndex):
            index = pandas_object
        else:
            raise RuntimeError(f'Input is not a df or a df index')

        time_indices = index.levels[0].to_list()
        space_indices = index.levels[1].to_list()

        index_tuples = index.to_list()

        time_space_indices = cls(time_indices=time_indices, space_indices=space_indices,
                                 index_tuples=index_tuples, ntime=len(time_indices),
                                 nspace=len(space_indices), nrow=len(index_tuples))

        return time_space_indices


def is_strideable(pandas_object):

    """
    is_strideable

    Function which accepts a pandas df or multindex and determines if data indexed with that index
    can be strided.

    """

    time_space = TimeSpaceIndices.from_pandas(pandas_object)

    if time_space.nrow == time_space.ntime * time_space.nspace:
        return True
    else:
        return False


def check_df_data_types(df):
    """
    check_df_data_types

    Check that there is only one data type in the input dataframe, and that it is in the set of allowed
    data types
    """

    dtypes_set = set(df.dtypes)

    if len(dtypes_set) != 1:
        raise RuntimeError(f'df with multiple dtypes passed: {df.dtypes}')

    dtype = list(dtypes_set)[0]

    if dtype not in defaults.allowed_dtypes:
        raise RuntimeError(f'dtype {dtype} not in allowed dtypes: {defaults.allowed_dtypes}')

    return dtype


def check_default_dtypes():
    """
    check_default_dtypes

    Check that the default datatypes defined in defaults can be handled by the rest of the package

    """

    if defaults.float_type not in defaults.allowed_dtypes:
        raise RuntimeError(f'default dtype {defaults.float_type} not in allowed dtypes: {defaults.allowed_dtypes}')
    if defaults.string_type not in defaults.allowed_dtypes:
        raise RuntimeError(f'default dtype {defaults.string_type} not in allowed dtypes: {defaults.allowed_dtypes}')


def get_dne(df):
    """
    get_dne

    Obtain correct does-not-exist token based on data type of input dataframe
    """

    dtype = check_df_data_types(df)

    if dtype in defaults.allowed_float_types:
        return defaults.fdne
    else:
        return defaults.sdne


def get_missing(df):
    """
    get_missing

    Obtain correct missing token based on data type of input dataframe
    """

    dtype = check_df_data_types(df)

    if dtype in defaults.allowed_float_types:
        return defaults.fmissing
    else:
        return defaults.smissing


def get_dtype(df):
    """
    get_dtype

    Obtain correct output datatype based on data type of input dataframe
    """

    dtype = check_df_data_types(df)
    check_default_dtypes()
    if dtype in defaults.allowed_float_types:
        return defaults.float_type
    else:
        return defaults.string_type


def df_to_numpy_time_space_strided(df):

    """
    df_to_numpy_time_space_strided

    Convert panel dataframe to numpy time-space-feature tensor using stride-tricks

    """

    dtype = get_dtype(df)

    # get shape of dataframe

    dim0, dim1 = df.index.levshape

    dim2 = df.shape[1]

    # check that df can in principle be tensorised

    if dim0 * dim1 != df.shape[0]:
        raise Exception("df cannot be cast to a tensor - dim0 * dim1 != df.shape[0]",
                        dim0, dim1, df.shape[0])

    flat = df.to_numpy()

    # get strides (in bytes) of flat array
    flat_strides = flat.strides

    offset2 = flat_strides[1]

    offset1 = flat_strides[0]

    # compute stride in bytes along dimension 0
    offset0 = dim1 * offset1

    # get memory view or copy as a numpy array
    tensor_time_space = np.lib.stride_tricks.as_strided(flat, shape=(dim0, dim1, dim2),
                                                        strides=(offset0, offset1, offset2))

    return tensor_time_space.astype(dtype)


def df_to_numpy_time_space_unstrided(df):
    """
    df_to_numpy_time_space_unstrided

    Convert panel dataframe to numpy time-space-feature tensor without using stride-tricks
    (for panels which are not simply-tensorisable)

    """

    dne = get_dne(df)
    dtype = get_dtype(df)

    time_space = TimeSpaceIndices.from_pandas(df)

    nfeature = len(df.columns)

    if df[df == defaults.sdne].sum().sum() > 0:
        raise RuntimeError(f'Default does-not-exist token {dne} found in input data')

    tensor_time_space = np.full((time_space.ntime, time_space.nspace, nfeature), dne, dtype=dtype)

    for irow in range(time_space.nrow):
        idx = time_space.index_tuples[irow]
        itime = time_space.time_indices.index(idx[0])
        ispace = time_space.space_indices.index(idx[1])
        tensor_time_space[itime, ispace, :] = df.values[irow]

    return tensor_time_space


def numpy_time_space_to_longlat(tensor_time_space, pandas_object):
    """
    numpy time_space_to_longlat

    Convert numpy time-space-feature tensor to a longitude-latitude-time-space tensor using
    stride-tricks
    """

    dtype = tensor_time_space.dtype

    dne = defaults.fdne if dtype in defaults.allowed_float_types else defaults.sdne

    time_units = TimeUnits.from_pandas(pandas_object)
    longlat_units = LonglatUnits.from_pandas(pandas_object)

    # convert 3d tensor into longitude x latitude x time x feature tensor

    tensor_longlat = np.full((longlat_units.gridsize,
                             longlat_units.gridsize,
                             len(time_units.times),
                             tensor_time_space.shape[-1]),
                             dne,
                             dtype=dtype)

    for pgid in longlat_units.pgids:

        pgindex = longlat_units.pgid_to_index[pgid]
        for time in time_units.times:
            tindex = time_units.time_to_index[time]
            ilong = longlat_units.pgid_to_longlat[pgid][0]
            ilat = longlat_units.pgid_to_longlat[pgid][1]

            tensor_longlat[ilong, ilat, tindex, :] = tensor_time_space[tindex, pgindex, :]

    return tensor_longlat


def time_space_to_panel_unstrided(tensor, index, columns):

    """
    time_space_to_panel_unstrided

    Convert numpy time-space-feature tensor to dataframe without using stride-tricks
    """

    dtype = tensor.dtype

    dne = defaults.fdne if dtype == np.float64 else len(max(tensor, key=len))*'-'

    time_space = TimeSpaceIndices.from_pandas(index)

    nfeature = tensor.shape[-1]

    data = np.full((time_space.nrow, nfeature), dne)

    for irow, row in enumerate(time_space.index_tuples):
        idx = time_space.index_tuples[irow]
        itime = time_space.time_indices.index(idx[0])
        ispace = time_space.space_indices.index(idx[1])
        data[irow, :] = tensor[itime, ispace, :]

    return pd.DataFrame(data=data, index=index, columns=columns)


def time_space_to_panel_strided(tensor, index, columns):
    """
    time_space_to_panel_strided

    Convert numpy time-space-feature tensor to dataframe using stride-tricks
    """

    time_space = TimeSpaceIndices.from_pandas(index)

    nfeature = tensor.shape[-1]

    tensor_strides = tensor.strides

    offset2 = tensor_strides[2]

    offset1 = tensor_strides[1]

    flat = np.lib.stride_tricks.as_strided(tensor, shape=(time_space.ntime * time_space.nspace, nfeature),
                                           strides=(offset1, offset2))

    return pd.DataFrame(flat, index=index, columns=columns)
