import cv2
import numpy as np

from ditto.shape_generator.shape import Shape
class Ellipse(Shape):
  def __init__(self, center, axeslength,angle, h, mask):
    super(Ellipse, self).__init__(mask)
    self.center = center
    self.axeslength = axeslength
    self.angle = angle
    self.h = h

  def move(self, offset):
    self.center = self.center + offset

  def rotate(self, angle):
    self.angle = self.angle + angle

  def draw(self, img:np.ndarray, draw_over=True):
    mask = np.zeros(img.shape)
    center = self.center.astype(int)
    axes = self.axeslength.astype(int)
    cv2.ellipse(mask, (center[0],center[1]), (axes[0],axes[1]), self.angle ,0,360,1, -1)
    # cv2.drawContours(mask, [self.get_global_polygon_pos()], 0, 1, -1)
    self.mask = mask * self.h
    if draw_over:
      img = img * (self.mask == 0) + self.mask
    else:
      img += self.mask
    assert isinstance(img, np.ndarray)
    return img
