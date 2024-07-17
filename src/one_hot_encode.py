from typing import Tuple, Union
import pandas as pd
import numpy as np
import sys
################################################################################
#---------------------------Custom Lib imports----------------------------------
################################################################################
# change path to src so you can import modules
sys.path.append('/content/drive/MyDrive/Academic/NYCDSA/Project-IV (Chip Pattern CNN)/src')
from resize_images import resize_images
################################################################################
#--------------------------One Hot Encode Maps Function-------------------------
################################################################################
# This function first one hot encodes maps with 0, 1, 2 values into 3 diff maps.
# NOTE 1: If you don't define a target height and width, then it won't resize the
# maps prior to encoding, if you don't resize the maps prior to encoding
# then it will return an array of array's, which is NOT usable for tensorflow!
# Therefore, you need to either always specify a height and width or resize
# the maps/images prior to feeding them into this function.
# NOTE 2: The images input need to be a list or series, so if you have np.array
# as input you need to mylist.append(myarray) prior to feeding it as input.
def one_hot_encode(images: Union[np.ndarray, pd.Series], 
                  map_size: Tuple[int,int] = (None,None) ) -> np.ndarray:
  # Checks for images correct type
  if not (isinstance(images, np.ndarray) or isinstance(images, pd.Series)):
    raise TypeError('Images input must be np.array or pd.Series')
  # Check if map_size input is Tuple[int, int]
  if not map_size==(None,None):
    if isinstance(map_size, tuple):
      if not all(isinstance(x, int) for x in map_size):
        raise TypeError('map_size must be tuple(int.int)')
    else:
      raise TypeError('map_size must be tuple(int.int)') 
  # Checks that the input maps are equal shape if user doesn't specify map_size
  if map_size==(None,None):
    for i in range(images.shape[0]):
      rows = images[0].shape[0]
      for j in range(2):
        if not (images[i].shape[j] == rows):
          raise ValueError("All maps must have the same dimensions, \n\
  you can resize maps with the map_size=(int,int) arg.")
  
  one_hot_maps = []
  # This resized the input images if a map_size is given, else
  # it will do nothing.
  resized_maps = resize_images(images, map_size) \
                if (map_size[0] and map_size[1]) else images
  for j in range(resized_maps.shape[0]):
    # One hot encoding: Using fancy compare you can compare an entire map at
    # once (images[j] == i), since it'll give a boolean output you need to
    # convert to int after (.astype(int)), lastly transpose((1,2,0)) changes the
    # dimensions so that you have (nth_map,height,width) instead of (height,width,nth_map)
    encoded_map = np.array([(resized_maps[j] == i).astype(int) for i in range(3)])\
    .transpose((1,2,0))
    one_hot_maps.append(encoded_map)
  # We convert to np.array before returning results since tensorflow needs
  # np.array's as input to work.
  return np.stack(one_hot_maps, axis=0)