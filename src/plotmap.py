import matplotlib.pyplot as plt
import numpy as np

# # NOTE: Input map must be a NxM np.array
# def plot_map(map: np.ndarray, title: str = None) -> None:
#   fig, ax = plt.subplots(figsize=(2,2))
#   ax.imshow(map)
#   ax.set_title(title,fontsize=10) if title else \
#   ax.set_title(str(map.shape),fontsize=10)
#   ax.set_xticks([])
#   ax.set_yticks([])
#   plt.tight_layout()
#   plt.show()

# NOTE: Input map must be a NxM np.array
def plot_map(map, title=None) -> None:
  fig, ax = plt.subplots(figsize=(2,2))
  ax.imshow(map)
  ax.set_title(title,fontsize=10) if title else \
  ax.set_title(str(map.shape),fontsize=10)
  ax.set_xticks([])
  ax.set_yticks([])
  plt.tight_layout()
  plt.show()