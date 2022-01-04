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
    X, Y = np.ogrid[:W, :H]
    mask = np.sqrt((1/long_axis * (X - c_w))**2 + (1/short_axis * (Y - c_h))**2) <= r

    angle=0
    if self.random_rotate:
      angle = np.random.randint(0, 360)
      # # 2 * N
      # index = np.where(mask==1)
      # index -= np.array([[c_w],[c_h]])
      # theta = np.random.uniform(0, np.pi)
      # rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)],
      #                           [np.sin(theta), np.cos(theta)]])
      # index = np.dot(rotate_matrix, index)
      # index += np.array([[c_w ],[c_h]])
      # index = index.astype(int)
      #
      # col_max, col_min = np.amax(index[0]), np.amin(index[0])
      # new_mask = np.zeros((W, H)).astype(int)
      # for i in range(col_min, col_max + 1):
      #   col_index = index[1, (index[0] == i)]
      #   row_max, row_min = np.amax(col_index), np.amin(col_index)
      #   new_mask[i, max(row_min,0):min(row_max+1, H)] = 1

      # mask = new_mask.astype(bool)

    h = np.random.uniform(self.height_range[0], self.height_range[1])

    self.counter += 1
    return Ellipse(np.array([c_w,c_h]), np.array([long_axis * r, short_axis* r]),
                   angle, h, mask)
    # return Shape(mask * h)

  def generate(self, img):
    shapes = []
    for _ in range(self.num):
      shapes.append(self.generate_one(img))

    return shapes
