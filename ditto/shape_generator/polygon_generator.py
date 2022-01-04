import math
import cv2
import numpy as np

from ditto.shape_generator import Shape_generator
from ditto.shape_generator.shape import Shape, Polygon

class Polygon_generator(Shape_generator):
  def __init__(self, radius_range, num_range, num_point_range, height_range, random_rotate=True,
               irregular=False):
    super(Polygon_generator, self).__init__()
    self.radius_range = radius_range
    self.num_range = num_range
    self.height_range = height_range
    self.random_rotate = random_rotate
    self.num = np.random.randint(self.num_range[0], self.num_range[1])
    self.num_point_range = num_point_range
    self.irregular = irregular

  def generate_one(self, img:np.array):
    H, W = img.shape[1], img.shape[0]
    r = np.random.uniform(self.radius_range[0] * W, self.radius_range[1] * W)
    c_w = np.random.randint(r, W - r)
    c_h = np.random.randint(r, H - r)
    X, Y = np.ogrid[:W, :H]
    mask = np.logical_and(np.sqrt((X - c_w)**2 + (Y - c_h)**2) <= self.radius_range[1] * W,
                          np.sqrt((X - c_w)**2 + (Y - c_h)**2) >= self.radius_range[0] * W - 1)
    index = np.transpose(np.where(mask))
    index = np.array(sorted(index, key=lambda x: math.atan2((x[1] - np.mean(index[:, 1])), (x[0] - np.mean(index[:, 0])))))
    num_points = np.random.randint(self.num_point_range[0], self.num_point_range[1])
    polygon_index = (np.linspace(0, len(index), num=num_points + 1)).astype(int)
    if self.irregular:
      for i in range(len(polygon_index)):
        polygon_index[i] += np.random.randint(len(index)//num_points)
    if self.random_rotate:
      polygon_index += np.random.randint(len(index))
      polygon_index = polygon_index % len(index)
    polygon = index[polygon_index[:-1]]
    mask = np.zeros((W, H))
    cv2.drawContours(mask, [polygon], 0, 1, -1)

    h = np.random.uniform(self.height_range[0], self.height_range[1])
    self.counter += 1
    return Polygon(polygon, h, mask)
    return Shape(mask * h)



  def generate(self, img):
    shapes = []
    for _ in range(self.num):
      shapes.append(self.generate_one(img))

    return shapes
