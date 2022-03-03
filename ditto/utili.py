import numpy as np



def parse(x):
  if isinstance(x, (tuple, list)):
    x = np.random.uniform(*x)
  return x

def rotation_matrix(angle):
  return np.array([[np.cos(angle), np.sin(angle)],
                   [-np.sin(angle), np.cos(angle)]])