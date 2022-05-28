from ditto.aberration_generator.aberration_generator import AbberationGenerator
import numpy as np
import cv2



class KaiqiangAberrationGenerator(AbberationGenerator):
  def __init__(self, config):
    if 'kaiqiang_aberration' in list(config.keys()):
      self.config = config['kaiqiang_aberration']
    else:
      self.config = config
    # self.random_rotate = self.config['random_rotate']

    self.grid_num_range = self.config['grid_num_range']
    # self.short_axis_range = self.config['short_axis_range']

    self.height_range = self.config['height_range']
    # self.radius_range = self.config['radius_range']

  def generate(self, img):
    W, H = img_size = img.shape
    grid_num = np.random.randint(*self.grid_num_range)
    random_number = np.random.randint(low=self.height_range[0],
                                      high=self.height_range[1],
                                      size=grid_num * grid_num)
    random_grid = np.reshape(random_number, (grid_num, grid_num)).astype('uint8')
    aberration = cv2.resize(random_grid, (H, W), interpolation=cv2.INTER_CUBIC).astype('float64')

    aberration += img

    return aberration
