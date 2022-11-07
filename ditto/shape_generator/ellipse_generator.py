from ditto.shape_generator import BaseGenerator
from ditto.shape_generator.shape import Shape, Ellipse
import numpy as np
import skimage



class EllipseGenerator(BaseGenerator):
  _name = 'Ellipse'

  def __init__(self, radius_range, long_axis_range, short_axis_range,
               num_range, height_range, random_rotate, center, uniform_height):
    super(EllipseGenerator, self).__init__()
    self.radius_range = radius_range
    self.long_axis = long_axis_range
    self.short_axis = short_axis_range
    self.num_range = num_range
    self.height_range = height_range
    self.random_rotate=random_rotate
    self.num = np.random.randint(self.num_range[0], self.num_range[1])
    self.center = center
    self.uniform_height = uniform_height

  def generate_one(self, img:np.array):
    H, W = img.shape[0], img.shape[1]
    radius_ratio = np.random.uniform(self.radius_range[0], self.radius_range[1])
    r = radius_ratio * np.min((H, W))
    long_axis = self.get_random_value(self.long_axis)
    short_axis = self.get_random_value(self.short_axis)
    if self.center:
      c_w = int(0.5 * W)
      c_h = int(0.5 * H)
    else:
      c_w = np.random.randint(0, W)
      c_h = np.random.randint(0, H)

    angle=0
    if self.random_rotate is not None:
      angle = self.get_random_value(self.random_rotate)

    h = self.get_random_value(self.height_range)
    self.counter += 1
    return Ellipse(np.array([c_w,c_h]), np.array([long_axis, short_axis]), r,
                   radius_ratio, angle, h, self.uniform_height)

  def generate(self, img):
    shapes = []
    for _ in range(self.num):
      shapes.append(self.generate_one(img))

    return shapes

EllipseGenerator.register()