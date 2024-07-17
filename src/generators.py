from typing import Tuple, Union
################################################################################
#----------------------------Test Data Generator--------------------------------
################################################################################
# NOTE: You can define a map_size to resize the images if you want, the default
# is (64,64)
# NOTE: This test_gen is NOT infinite while the validate_gen is.
def test_gen(input_df: pd.DataFrame, 
             batch_size: int = 100, 
             map_size: Tuple[int,int] = (64,64),
             classes: int = None):
  
  if not isinstance(input_df, pd.DataFrame):
    raise TypeError("input_df must be a pandas DataFrame")
  if not isinstance(batch_size, int):
    raise TypeError("batch_size must be an int")
  if not isinstance(map_size, tuple):
    raise TypeError("map_size must be a tuple")
  if not isinstance(classes, int):
    raise TypeError("classes must be an int")

  x_test, y_test = [], []
                    
  if classes is None:
    # Finding the total number of fail mode within the df
    classes = input_df['failurenum'].max()+1
  else:
    classes = classes
 
  # Sampling batch_size of data from the the Input
  test = input_df.sample(n=batch_size, replace=False)#, random_state=1)
  test.reset_index(inplace = True, drop = True)# reseting index inplace
                    
  # Resize maps
  resized_maps = resize_images(test['wafermap'], map_size)

  # One hot encode maps, this returns a np.array and also resizes maps.
  encoded_maps = one_hot_encode(resized_maps, map_size )

  # Creates an array of dim: [batch_size] X ([classes]) of all zeros
  labels = np.zeros((encoded_maps.shape[0], classes))

  # fills in only the column corresponding to the failNum with "1"
  for i in range(encoded_maps.shape[0]):
      labels[i][test['failurenum'][i]] = 1
  del test
                    
  x_test.extend(encoded_maps)
  y_test.extend(labels)
  x_test = np.array(x_test)
  y_test = np.array(y_test)
  # display("there are ",len(x_test),"maps for testing")
  # num += 1
  yield(x_test, y_test)