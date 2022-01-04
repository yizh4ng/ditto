from ditto.aberration_generator.abberation_generator import AbberationGenerator
import numpy as np


class TiltAbberationGenerator(AbberationGenerator):
  def __init__(self, slope_direction, slope, lowest_height=0):
    self.slop_direction = slope_direction
    self.slop = slope
    self.lowest_height = lowest_height

  def generate(self, img):
    W, H = img.shape[0], img.shape[1]
    tilt = np.zeros((W, H))

    for w in range(W):
      for h in range(H):
        tilt[w][h] = np.dot([w, h], self.slop_direction) * self.slop

    return img + tilt + self.lowest_height


