import imgaug as ia
import imgaug.augmenters as iaa
import pandas as pd
import numpy as np
################################################################################
#---------------------Defining the augmenter model/function---------------------
################################################################################
seq = iaa.Sequential([
    # iaa.Fliplr(1), # horizontally flip 50% of the images
    # Apply affine transformations to some of the images
    iaa.Affine(
        # - scale to 80-120% of image height/width (each axis independently)
        # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        # Translate images by a certain percent
        # "x": (- Left translation, +right translation)
        # "y": (- Left translation, +right translation)
        # translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        # rotate randomly by 90,-90, -180 or +180 degrees
        rotate=(np.random.choice([90,-90,180,-180])),
        # shear by -X to +X degrees, it will randomly choose a choose a degree
        # of shearing between these two numbers.
        # shear=(-30,30)
    ),
], random_order=True) # apply augmenters in random order
################################################################################
#--------------------------Augmenting Maps Function-----------------------------
################################################################################
def augment_images(images: pd.Series, number: int = None) -> np.ndarray:
    # Check if images input is pd.series
    if not isinstance(images, pd.Series):
      raise TypeError('images must be pd.Series')
      # Check if images input is pd.series
    if number != None:
      if not isinstance(number, int):
        raise TypeError('number must be int')
    # Randomly choose "number" maps from the "images" input if number is given
    # as input, if no number is given as input then it will return the entire
    # images array. This means it will apply the augmentation to the entire
    # input array, or you can apply to subset only "number".
    images_input = np.random.choice(images, number) if number else images
    # Converts from pd.Series(n,) to np.array(n,N,M) where each map is NxM
    # This is needed since 'seq' only accepts np.array as input.
    images_input = np.stack(images_input.to_numpy())
    images_augmented = seq(images=images_input)
    return images_augmented