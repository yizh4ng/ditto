# generate an image
import random

import numpy as np
import matplotlib.pyplot as plt


from ditto.image.fringeencoder import FringeEncoder
from ditto.image.interferogram import Interferogram
from ditto.image.config import Config
from scipy.signal import convolve2d
from ditto.shape_generator import shape_dict
from ditto.aberration_generator import abberation_dict

class Painter(FringeEncoder):
  def __init__(self, img_size, **kwargs):
    img = np.zeros((img_size[0], img_size[1]))
    self.config = kwargs
    super(Painter, self).__init__(img, **kwargs)

  def paint_samples(self, refresh=True):
    if refresh:
      self.img=np.zeros((self.img.shape[0], self.img.shape[1]))
    # shapes_generator = []
    shapes = []
    draw_over = self.config['draw_over']
    for key in self.config.keys():
      if key in shape_dict.keys():
        # shapes_generator.append(shape_dict[key](**self.config[key]))
        shapes.extend(shape_dict[key](**self.config[key]).generate(self.img))


    if self.config['suffle']:
      random.shuffle(shapes)
    for s in shapes:
      self.img = s.draw(self.img, draw_over)

    self.shapes = shapes
    if self.config['smooth']:
      conv_size = 2*self.config['smooth']+1
      self.img = convolve2d(self.img,
                             (np.zeros([conv_size,conv_size]) + 1/(conv_size ** 2)),
                              boundary='symm',
                              mode='same')
    self.ground_truth = self.img

    for key in self.config.keys():
      if key in abberation_dict.keys():
        self.img = abberation_dict[key](*self.config[key].values()).generate(self.img)


if __name__ == '__main__':
  from lambo.gui.vinci.vinci import DaVinci
  p = Painter(**Config)
  p.paint_samples()
  da = DaVinci()
  da.objects = [p.ground_truth, p.img,
                np.log(np.abs(p.F) + 1),
                np.log(np.abs(p.masked) + 1),
                np.log(np.abs(p.uncentralized_signal) + 1),
                p.extracted_fringe]
  da.add_plotter(da.imshow)
  da.show()

  ig = Interferogram(img=-p.extracted_fringe, radius=30)
  ig.dashow()
  # p.paint_samples()
  # da = DaVinci()
  # da.objects = [p.img,
  #               np.log(np.abs(p.F) + 1),
  #               p.mask,
  #               np.log(np.abs(p.uncentralized_signal) + 1),
  #               p.extracted_fringe]
  # da.add_plotter(da.imshow)
  # da.show()
  # p = Painter(**Config)
  # p.paint_samples()
  # da = DaVinci()
  # da.objects = [p.img,
  #               np.log(np.abs(p.F) + 1),
  #               p.mask,
  #               np.log(np.abs(p.uncentralized_signal) + 1),
  #               p.extracted_fringe]
  # da.add_plotter(da.imshow)
  # da.show()
