from typing import Tuple
import pandas as pd
################################################################################
#------------------------------Split Test/Train Data----------------------------
################################################################################
def split(df: pd.DataFrame, 
          test_percent: float = 0.02) -> Tuple[pd.DataFrame,pd.DataFrame]:

  if not isinstance(test_percent, float):
    return TypeError("percent must be a float")
  elif test_percent < 0 or test_percent > 1:
    return ValueError("percent must be between 0 and 1")
  # finds fail pattern with min # of maps
  # min_pattern = df.groupby(['failurenum']).count().sort_values(by='failuretype').iloc[0,0]

  # Sample a given percent from each fail mode for the test_df
  # if test_percent:
  test_df = df.groupby(['failurenum'], group_keys = False)\
              .apply(lambda x: x.sample(frac=test_percent, random_state=1))
  # else:
  #   test_df = df.groupby(['failurenum'], group_keys = False)\
  #               .apply(lambda x: x.sample(n=min_pattern, random_state=1))
  # test_df.reset_index(inplace = True, drop = True)# reseting index inplace
  # Concat sampled df to original df
  train_df = pd.concat([test_df, df])
  # Use index difference to remove test_df data from label_pattern so that we don't
  # use test_df for the training (we are left with ~24.349 maps for training)
  train_df = train_df.loc[train_df.index.difference(test_df.index),]
  train_df.reset_index(inplace = True, drop = True)# reseting index inplace
  test_df.reset_index(inplace = True, drop = True)# reseting index inplace
  return test_df, train_df