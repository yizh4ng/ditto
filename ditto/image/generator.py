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
  def __init__(self, conifg, abberation_image=None):
    self.config = conifg
    img_size = self.config['img_size']
    img = np.zeros((img_size[0], img_size[1]))
    self.abberation_image = abberation_image
    super(Painter, self).__init__(img, self.config, abberation_image)

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
    self.ground_truth = np.copy(self.img)

    for key in self.config.keys():
      if key in abberation_dict.keys():
        self.img = abberation_dict[key](*self.config[key].values()).generate(self.img)

    # if self.abberation_image is not None:
    #   self.bg_ig = Interferogram(self.abberation_image,
    #                             radius=100)
    #   self.img += self.bg_ig.extracted_angle_unwrapped
      # self.abberation = self.bg_ig.low_frequency_filter
      # self.img += 0.01 * self.abberation
      # self.img *= self.abberation

if __name__ == '__main__':
  def normalize(img):
    img -= np.min(img)
    img = img / np.max(img)
    return img
  from lambo.gui.vinci.vinci import DaVinci
  # from lambo.gui.vincv.davincv import DaVincv
  from ditto import Painter, ImageConfig
  import cv2
  abberation_image = cv2.imread('G:/projects/data_old/01-3t3/bg/1.tif')[:,:,0]
  # abberation_image = cv2.imread('G:/projects/data_old/05-bead/100007.tif')[:,:,0]
  bg = cv2.imread('G:/projects/data_old/05-bead/100008.tif')[:,:,0]
  abberation_image_ = np.copy(abberation_image)
  p = Painter(ImageConfig, abberation_image)
  p.paint_samples()
  # p1 = Painter(ImageConfig)
  # bg = p1.physics_based_fringe
  da = DaVinci()
  fringe = p.physics_based_fringe
  abberation_image = p.abberation_image - p.bg_ig.low_frequency_filter
  mean = np.mean(abberation_image)
  abberation_image_copy = np.copy(abberation_image)
  # abberation_image[abberation_image_copy<mean] = 0
  # abberation_image[abberation_image_copy>mean] = -1
  abberation_image = normalize(abberation_image)
  da.objects = [p.ground_truth, p.img,
                # np.log(np.abs(p.F) + 1),
                # np.log(np.abs(p.masked) + 1),
                # np.log(np.abs(p.uncentralized_signal) + 1),
                # p.extracted_fringe,
                # abberation_image - p.bg_ig.low_frequency_filter,
                p.bg_ig.low_frequency_filter,
                p.bg_ig.high_frequency_filter,
                p.abberation_image,
                # abberation_image,
                fringe,
                # fringe-p.bg_ig.low_frequency_filter
                ]
  da.add_plotter(da.imshow)
  da.show()

  # ig = Interferogram(img=fringe, radius=120)
  # ig.dashow()
  ''''Following codes are for testing whether the backgrounds matter'''
  def binary(phase_map):
    mean = np.mean(phase_map)
    zero_indices = phase_map <= mean
    one_indices = phase_map > mean
    phase_map_ = np.zeros_like(phase_map)
    phase_map_[zero_indices] = 0
    phase_map_[one_indices] = 1
    return phase_map_
  fringe = binary(p.bg_ig.high_frequency_filter)
  ig = Interferogram(img=fringe, radius=120)
  phase_1 = ig.extracted_angle_unwrapped
  phase_1 = normalize(phase_1)
  ig = Interferogram(img=abberation_image_, radius=120, bg_array=bg)
  ig.dashow()
  phase_2 = ig.extracted_angle_unwrapped
  phase_2 = normalize(phase_2)
  da = DaVinci()
  da.objects = [abberation_image_, p.bg_ig.high_frequency_filter, fringe, phase_1, phase_2]
  da.add_plotter(da.imshow)
  da.show()
  print(np.mean(np.abs(phase_1 - phase_2)))

  '''Following codes are for testing the components of an Interferogram'''
  # ig = Interferogram(img=abberation_image_, radius=120)
  # low_frequency = ig.low_frequency_filter
  # high_frequency = ig.high_frequency_filter
  # ig_ = Interferogram(high_frequency, radius=120)
  # I2 = high_frequency / np.cos(ig_.extracted_angle)
  # da = DaVinci()
  # da.objects = [abberation_image_, low_frequency, high_frequency, low_frequency + high_frequency]
  # I1 = low_frequency
  # I2 = np.abs(ig_.extracted_image)
  # y = I1 + np.sqrt(np.square(I1) - np.square(I2))
  # x = I1 - y
  # da.objects = [abberation_image_, high_frequency, low_frequency, x, y]
  # da.add_plotter(da.imshow)
  # print(np.mean(np.abs(low_frequency + high_frequency - abberation_image_)))
  # da.show()
