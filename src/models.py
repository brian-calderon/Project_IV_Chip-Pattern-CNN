import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
################################################################################
#-------------------------------LeNet Model-------------------------------------
################################################################################
def lenet(input_shape = (None, None, None), classes: int = None):
  if not isinstance(input_shape, tuple):
    raise TypeError('input_shape must be a tuple.')
  if len(input_shape) != 3:
    raise ValueError('input_shape must be a tuple of length 3.')
  if not isinstance(classes, (int, np.int64)):
    raise TypeError('classes must be an integer.')
  if classes <= 2:
    raise ValueError('classes must be an integer greater than 2')
    
  inputs = keras.Input(shape=input_shape)
  x = layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D()(x) #Default is 2x2

  x = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D()(x) #Default is 2x2

  x = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D()(x) #Default is 2x2

  x = layers.Flatten()(x)
  x = layers.Dense(64, activation = 'relu')(x)

  x = layers.Dense(32, activation = 'relu')(x)

  outputs = layers.Dense(classes, activation = 'softmax')(x)

  model = keras.Model(inputs = inputs, outputs = outputs)

  return model