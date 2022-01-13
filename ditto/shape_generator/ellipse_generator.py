from ditto.shape_generator import Shape_generator
from ditto.shape_generator.shape import Shape, Ellipse
import numpy as np
import skimage

class Ellipse_generator(Shape_generator):
  def __init__(self, radius_range, long_axis_range, short_axis_range,
               num_range, height_range, random_rotate=True):
    super(Ellipse_generator, self).__init__()
    self.radius_range = radius_range
    self.long_axis = long_axis_range
    self.short_axis = short_axis_range
    self.num_range = num_range
    self.height_range = height_range
    self.random_rotate=random_rotate
    self.num = np.random.randint(self.num_range[0], self.num_range[1])

  def generate_one(self, img:np.array):
    H, W = img.shape[1], img.shape[0]
    r = np.random.uniform(self.radius_range[0] * W, self.radius_range[1] * W)
    long_axis = np.random.uniform(self.long_axis[0], self.long_axis[1])
    short_axis = np.random.uniform(self.short_axis[0], self.short_axis[1])
    c_w = np.random.randint(long_axis * r, W - long_axis * r)
    c_h = np.random.randint(long_axis * r, W - long_axis * r)

    angle=0
    if self.random_rotate:
      angle = np.random.randint(0, 360)

    h = np.random.uniform(self.height_range[0], self.height_range[1])

    self.counter += 1
    return Ellipse(np.array([c_w,c_h]), np.array([long_axis * r, short_axis* r]),
                   angle, h)
    # return Shape(mask * h)

  def generate(self, img):
    shapes = []
    for _ in range(self.num):
      shapes.append(self.generate_one(img))

    return shapes
