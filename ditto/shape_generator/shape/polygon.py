import cv2
import numpy as np
from ditto.shape_generator.shape import Shape



class Polygon(Shape):
  def __init__(self, polygon, h):
    super(Polygon, self).__init__()
    self.polygon = polygon
    self.h = h
    self.angle = 0

  def move(self, offset):
    self.polygon = self.polygon + offset

  def rotate(self, angle):
    self.angle = self.angle + angle

  def get_global_polygon_pos(self):
    polygon = np.transpose(self.polygon)
    c_w = np.mean(polygon[0,:]).astype(int)
    c_h = np.mean(polygon[1,:]).astype(int)
    polygon = polygon - np.array([[c_w], [c_h]])
    theta = self.angle / (2 * np.pi)
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    polygon = np.dot(rotate_matrix, polygon).astype(int)
    polygon = polygon + np.array([[c_w], [c_h]])
    polygon = np.transpose(polygon)
    return polygon

  def _get_mask(self, img):
    mask = np.zeros(img.shape)
    cv2.drawContours(mask, [self.get_global_polygon_pos()], 0, 1, -1)
    mask = mask * self.h
    return mask

  # def draw(self, img:np.ndarray, draw_over=True):
  #   mask = np.zeros(img.shape)
  #   cv2.drawContours(mask, [self.get_global_polygon_pos()], 0, 1, -1)
  #   self.mask = mask * self.h
  #   if draw_over:
  #     img = img * (self.mask == 0) + self.mask
  #   else:
  #     img += self.mask
  #   assert isinstance(img, np.ndarray)
  #   return img