from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union
# Supressing the chained assign warning
# pd.options.mode.chained_assignment = None

# Cleaning the dataset
@dataclass()
class cleanDF:
  _df: pd.DataFrame

  def __post_init__(self: cleanDF):
    # Make a copy of the DataFrame to avoid modifying the original
    self._df = self._df.copy()

  # lowercase cols, removes extra cols, adds wafermapdim and encode cols
  def clean(self: cleanDF) -> pd.DataFrame:
    # Convert column names to lower case
    self._df = self.columns_to_lowercase()
    # Get list of column names
    columns = self._df.columns.tolist()
    # Check if df has wafermap and failuretype columns
    if (sum([x=='wafermap' for x in columns]) + 
        sum([x=='failuretype' for x in columns])) == 2:
      pass
    else:
      raise NameError('df must contain two columns named [\'wafermap\',\'failuretype\']')

    # Check if all rows in wafermap column are of type np.array
    if self._df['wafermap'].apply(lambda map: isinstance(map, np.ndarray)).all():
      pass
    else:
      raise TypeError('rows in wafermap must be NxM np.arrays')

    # Check if failuretype is str type, if not convert
    if not isinstance(self._df['failuretype'],str):
      # This is very specific to the WMK811K dataset. Should remove for other stuff.
      # that data set has the failuretype col. set to a (1,1) np.array so we need to 
      # first change it to a 1d np.array
      self._df['failuretype'] = self._df['failuretype'].apply(np.squeeze)
      # Converts to str
      self._df['failuretype'] = self._df['failuretype'].astype('string')

    # Erase all other columns if they exist
    if self._df.shape[1]>2:
      # Boolean mask to match only columns we need
      keep_cols = [col=='wafermap' or col=='failuretype' for col in columns]
      # use bool mask to keep only important cols
      self._df=self._df[self._df.columns[keep_cols]]
    else:
      pass

    # Adding column with wafer map dimensions
    self._df = self.add_wafermap_dim()

    # Add encoding column
    self._df = cleanDF.encode(self._df)

    # Resetting index inplace
    self._df.reset_index(inplace = True, drop = True)
    return self._df

  @staticmethod
  # Adding encoding failureType column
  def encode(df: pd.DataFrame) -> pd.DataFrame:
    encoder={}
    # Gets alphabetically sorted fail mode names
    fail_modes=sorted(df['failuretype'].unique())
    for n in range(len(fail_modes)): 
      encoder[fail_modes[n]] = n
    # NOTE: You have change the column 'failureType' to type 'object' before you
    # an replace its string values with numeric. Object functions as multiple types,
    # whereas string can only be replaced with other strings.
    df['failurenum'] = df['failuretype'].astype(object).replace(encoder)
    return df

  # Adding column with wafer map dimensions
  def add_wafermap_dim(self: cleanDF) -> pd.DataFrame:
    columns = self._df.columns.tolist()
    if sum([x=='wafermap' for x in columns]) == 1:
      pass
    else:
      raise NameError('df must contain atleast one column named [\'wafermap\']')
    self._df.loc[:,'waferdim'] =self._df.loc[:,'wafermap'].apply(cleanDF.find_dim)
    return self._df
  

  # Convert column names to lower case
  def columns_to_lowercase(self: cleanDF) -> pd.DataFrame:
    lowercase_columns=[col.lower() for col in self._df.columns.tolist()]
    # Change column names to lowercase
    self._df.columns=lowercase_columns
    return self._df
  
  @staticmethod
  # Retrieves wafer map dimensions from wafer map column
  def find_dim(x: Union[np.ndarray,pd.Series]):
    if not isinstance(x, (np.ndarray,pd.Series)):
      raise TypeError('x must be a np.array or pd.Series')

    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return (dim0, dim1)