import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Union
################################################################################
#----------------------------Resize Maps Function-------------------------------
################################################################################
# NOTE: The images input need to be a pd.Series so that .apply works.
# It returns a np.array since tflow likes those as input.
def resize_images(images: pd.Series, map_size: Tuple[int, int] = (64,64)) -> np.ndarray:
  # Check if images input is pd.series
  if not isinstance(images, pd.Series):
    raise TypeError('Images input must be pd.series')
  # Check if map_size input is Tuple[int, int]
  if map_size:
    if isinstance(map_size, tuple):
      if not all(isinstance(x, int) for x in map_size):
        raise TypeError('map_size must be tuple(int.int)')
    else:
      raise TypeError('map_size must be tuple(int.int)') 
  # Resizing input images
  images = images.apply(lambda x: cv2.resize(x, dsize=map_size,\
                                                   interpolation=cv2.INTER_NEAREST))
  return images