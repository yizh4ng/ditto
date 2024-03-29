import math
import cv2
import numpy as np

from ditto.shape_generator import BaseGenerator
from ditto.shape_generator.shape import Shape, Polygon

class PolygonGenerator(BaseGenerator):
  _name = 'Polygon'

  def __init__(self, radius_range, num_range, num_point_range, height_range, random_rotate=True,
               irregular=False):
    super(PolygonGenerator, self).__init__()
    self.radius_range = radius_range
    self.num_range = num_range
    self.height_range = height_range
    self.random_rotate = random_rotate
    self.num = np.random.randint(self.num_range[0], self.num_range[1])
    self.num_point_range = num_point_range
    self.irregular = irregular

  def generate_one(self, img:np.array):
    H, W = img.shape[0], img.shape[1]
    min_radius = self.radius_range[0] * min(W, H)
    max_radius = self.radius_range[1] * max((W, H))
    c_w = np.random.randint(max_radius, W - max_radius)
    c_h = np.random.randint(max_radius, H - max_radius)
    num_points = np.random.randint(self.num_point_range[0], self.num_point_range[1])
    def rotation_matrix(a):
      return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    polygon = []
    theta = 0
    delta = 2 * np.pi / num_points
    if self.random_rotate:
      theta += np.random.randint(0, 360)
    for i in range(num_points):
      angle = theta
      if self.irregular:
        angle += np.random.uniform(0, delta)
      polygon.append(np.array([c_w, c_h]) + np.dot(rotation_matrix(angle),
                     [np.random.uniform(min_radius, max_radius), 0]))
      theta += delta


    polygon = np.array(polygon).astype(int)
    mask = np.zeros((W, H))
    cv2.drawContours(mask, [polygon], 0, 1, -1)

    h = self.get_random_height()
    self.counter += 1
    return Polygon(polygon, h)



  def generate(self, img):
    shapes = []
    for _ in range(self.num):
      shapes.append(self.generate_one(img))

    return shapes

PolygonGenerator.register()