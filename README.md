# views_tensor_utilities

This package is a set of tools to allow users to transfer data in typical VIEWS format between pandas DataFrames and 
numpy arrays (referred to as tensors).

## VIEWS dataframes

VIEWS dataframes contain one or more features of panel data indexed by a two-column pandas MultiIndex. The first index 
column is a time unit (e.g. month, year) and the second is a spatial unit (most commonly country or priogrid cell). 
Missing data is represented by NaNs.

Months currently range from 1 (Jan 1980) to 852 (December 2050). There is no month 0. Countries and priogrid cells
are denoted by non-consecutive integers.

A crucial difference between the country and priogrid spatial units is that the definition of priogrid is fixed in 
time, so all cells exist for all time units, but **this is not the case for countries** - countries sometimes cease
to exist, or come into existence during the temporal range of a dataset. In a VIEWS dataframe, this is trivially 
represented by omitting the relevant (time-unit, space-unit) units of analysis.

Pandas dataframes are able to store numerical and string data in the same panel.

Pandas dataframes are able to store strings giving names to the index columns and feature columns.

## VIEWS tensors

VIEWS tensors come in two forms. For the purposes of regression models, 3-dimensional tensors are used, with the 
dimensions being (time-unit, space-unit, feature). For neural-net-based models and visualisation, 4-dimensional 
tensors with dimensions (longitude-unit, latitude-unit, time-unit, feature) are used.

Tensor indices are contiguous sets of integers starting from 0.

Ordinary Numpy arrays cannot be used to store mixed numeric and string data.

Ordinary Numpy arrays do not store names for their axes or axis values.

## Representing VIEWS dataframes as tensors

The _views_tensor_utilities_ package represents dataframes by wrappers around pure numpy arrays, accompanied by 
minimal metadata to capture essential information from the dataframe which cannot be stores in these arrays, 
such that it is possible to reconstruct the original dataframe (possibly with some reordering of the columns). 
Each ViewsNumpy object holds a tensor containing **only** numerical data or **only** string data, the column 
names from the original dataframe corresponding to the data stored in the tensor, and the values of the tokens 
used to represent missing data and non-existent units of analysis.

### The ViewsContainer class

An entire dataframe is represented by the ViewsTensorContainer class. This holds 
- a list of ViewsNumpy objects 
- the index of the original dataframe

The methods belonging to this class are

- _to_pandas_: guard method which checks whether the container tensors are 3D (which can be converted back to a 
  datafame) or 4D (which currently cannot) and respectively executes the conversion or returns an error
- _space_time_to_panel_: method which converts the containers tensors to dataframes, combines them into a single
  dataframe and returns it
- get_numeric_tensor: convenience method which retrieves the numeric tensor component, if it exists.
- get_string_tensor: convenience method which retrieves the string tensor component, if it exists.

### The ViewsNumpy class

This is a simple wrapper for a single numpy tensor containing **either** numeric **or** string data. It holds
- a numpy array representing a 3D time-space-feature tensor or a 4d longitude-latitude-time-feature tensor
- a list of columns names corresponding to the indices of the tensor's last (i.e. 3rd or 4th) dimension
- a value for the does-not-exist token used to denote units-of-analysis that do not exist
- a value for the missing token denoting legal units-of-analysis with undefined values

This class has no methods.

## The ViewsDataframe class

This class holds

- a pandas dataframe
- the index of the pandas dataframe
- a list of dataframes formed by splitting the original dataframe into numeric and string portions
- a transformer function, selected according to whether the input dataframe is strideable

The methods belonging to this class are

- __split_by_dtype: protected method which splits the input dataframe into 'number' and 'object' dataframes
  according to its column datatypes. If this fails to assign all the original dataframes to one of the 
  split dataframes, an error is raised
- to_numpy_time_space: this method calls __split_by_dtype, casts the split dataframes into (time, space,
  feature) tensors. These are wrapped into ViewsNumpy objects which also store the column names belonging to 
  the split tensors, and the does-not-exist and missing tokens used in building the tensors. The ViewsNumpy 
  objects and the original dataframe index are packed into a ViewsTensorContainer object which is returned.
- to_numpy_longlat: this method first calls to_numpy_time_space, then casts the (time, space, feature)
  tensors to (longitude, latitude, time, feature) tensors, before returning the modified ViewsTensorContainer.

# Examples

## Converting a VIEWS dataframe into tensors

This is done by instantiating the ViewsDataframe class

```
views_dataframe = objects.ViewsDataframe(df)
```
The command
```
tensor_container=views_dataframe.to_numpy_time_space()
```
generates a tensor container containing one of more ViewsNumpy objects wrapping the numeric and/or string
portions of the dataframe's data. These can be accessed by 
```
tensor=tensor_container.ViewsTensors[0].tensor
```
Alternatively, the convenience methods can be used
```
tensor=tensor_container.get_numeric_tensor()
```
or
```
tensor=tensor_container.get_string_tensor()
```

## Converting a ViewsTensorContainer into a pandas DataFrame

This is done by calling the container's to_pandas method:

```
df=tensor_container.to_pandas()
```
Note that the columns in the regenerated dataframe will likely not be in the same order as in the original
dataframe.