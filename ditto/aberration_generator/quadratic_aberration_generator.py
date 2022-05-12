from ditto.aberration_generator.aberration_generator import AbberationGenerator
import numpy as np
from scipy.spatial.transform import Rotation as R



class QuadraticAberrationGenerator(AbberationGenerator):
  def __init__(self, config):
    if 'quadratic_aberration' in list(config.keys()):
      self.config = config['quadratic_aberration']
    else:
      self.config = config
    self.random_rotate = self.config['random_rotate']

    self.long_axis_range = self.config['long_axis_range']
    self.short_axis_range = self.config['short_axis_range']

    self.height_range = self.config['height_range']
    self.radius_range = self.config['radius_range']

  def generate(self, img):
    W, H = img_size = img.shape

    r = R.from_euler('x', np.random.randint(0, 365), degrees=True)
    r_matrix = r.as_matrix()[1:, 1:]

    center_x, center_y = np.random.uniform(0, H), np.random.uniform(0, W)
    # center_x, center_y = 0.5 * H, 0.5 * W

    X_cor_range, Y_cor_range = np.arange(W), np.arange(H)
    Xv, Yv = np.meshgrid(Y_cor_range, X_cor_range)
    cor = np.transpose(np.array([Xv, Yv]), axes=(1, 2, 0))
    X, Y = cor[:,:,0], cor[:,:,1]

    if self.random_rotate:
      new_cor = np.sum(r_matrix @ np.expand_dims(
        np.transpose(np.array([Xv - center_x, Yv - center_y]), axes=(1, 2, 0)), -1), -1)
      X, Y = new_cor[:, :, 0] + center_x, new_cor[:, :, 1] + center_y

    radius = np.random.uniform(*self.radius_range)
    long_axis = np.random.uniform(*self.long_axis_range)
    short_axis = np.random.uniform(*self.short_axis_range)
    # long_axis, short_axis = 1, 1
    aberration = np.sqrt(
      radius**2 * (H**2 + W**2) - (long_axis * (X - center_x)) ** 2
      - (short_axis *(Y - center_y)) ** 2)
    height = np.random.uniform(*self.height_range)
    aberration /= (np.max(aberration) - np.min(aberration)) / height

    return img + aberration
