import itertools

import cv2
import numpy as np

from ditto.shape_generator.shape import Shape
class Ellipse(Shape):
  def __init__(self, center, axeslength, angle, h):
    super(Ellipse, self).__init__()
    self.center = center
    self.axeslength = axeslength
    self.angle = angle
    self.h = h

  def move(self, offset):
    self.center = self.center + offset

  def rotate(self, angle):
    self.angle = self.angle + angle

  def _get_mask(self, img):

    mask = np.zeros(img.shape)
    center = self.center.astype(int)
    axes = self.axeslength.astype(int)

    for long in range(int(self.axeslength[0] + 1)):
      short = self.axeslength[1] / self.axeslength[0] * long
      long_ratio = long / self.axeslength[0]
      h = self.h * (np.sqrt(1 - long_ratio ** 2))
      cv2.ellipse(mask, (center[0], center[1]), (int(long), int(short)),
                  self.angle, 0, 360, h, 2)
    # cv2.ellipse(mask, (center[0], center[1]), (axes[0], axes[1]), self.angle,
    #             0, 360, self.h, -1)
    return mask

  # def draw(self, img:np.ndarray, draw_over=True):
  #   mask = np.zeros(img.shape)
  #   center = self.center.astype(int)
  #   axes = self.axeslength.astype(int)
  #
  #   for long in range(int(self.axeslength[0]+1)):
  #     short = self.axeslength[1]/self.axeslength[0] * long
  #     long_ratio = long/self.axeslength[0]
  #     h =  self.h * (np.sqrt(1 -long_ratio**2))
  #     cv2.ellipse(mask, (center[0],center[1]), (int(long), int(short)),
  #                 self.angle,0,360, h, 2)
  #   # cv2.ellipse(mask, (center[0], center[1]), (axes[0], axes[1]), self.angle,
  #   #             0, 360, self.h, -1)
  #   self.mask = mask
  #   if draw_over:
  #     img = img * (self.mask == 0) + self.mask
  #   else:
  #     img += self.mask
  #   assert isinstance(img, np.ndarray)
  #   return img
