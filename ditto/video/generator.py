from ditto import Painter
import numpy as np

from ditto.aberration_generator import abberation_dict
from ditto.video.config import Config as VideoConfig
from scipy.signal import convolve2d

class VideoGenerator():
  def __init__(self, ImageConfig, VideoConfig):
    self.painter = Painter(ImageConfig)
    self.img_config = ImageConfig
    self.img_size = self.painter.img.shape
    self.video_length = VideoConfig['video_length']
    self.painter.paint_samples()
    self.first_frame = self.painter.img
    self.img_stack = [self.first_frame]
    self.fringe_stack = [self.painter.physics_based_fringe]
    self.shapes = self.painter.shapes
    self.move_range = VideoConfig['move_range']
    self.rotate_range = VideoConfig['rotate_range']
    self.back_ground = np.zeros(self.img_size)
    for key in self.img_config.keys():
      if key in abberation_dict.keys():
        self.back_ground = abberation_dict[key](*self.img_config[key].values()).generate(self.back_ground)

  def step(self):
    W, H = self.img_size[0], self.img_size[1]
    img = np.zeros((W, H))
    for s in self.shapes:

      s.move(offset=(np.random.randint(*self.move_range[0]),
                     np.random.randint(*self.move_range[1])))
      s.rotate(angle=np.random.randint(*self.rotate_range))
      img = s.draw(img, draw_over=self.img_config['draw_over'])
    if self.img_config['smooth']:

      conv_size = 2 * self.img_config['smooth'] + 1
      img = convolve2d(img,
                          (np.zeros([conv_size, conv_size]) + 1 / (
                              conv_size ** 2)),
                          boundary='symm',
                          mode='same')
    ground_turth = img
    # self.painter.img = img[W:-W, H:-H]
    img += self.back_ground

    self.painter.img = img
    self.img_stack.append(ground_turth)
    self.fringe_stack.append(self.painter.physics_based_fringe)

  def generate(self):
    for _ in range(self.video_length - 1):
      self.step()

if __name__ == '__main__':

  # from lambo.gui.vinci.vinci import DaVinci
  from lambo.gui.vincv.davincv import DaVincv
  import datetime
  starttime = datetime.datetime.now()
  from ditto import ImageConfig, VideoConfig, VideoGenerator
  vg = VideoGenerator(ImageConfig, VideoConfig)

  vg.generate()

  endtime = datetime.datetime.now()
  print((endtime - starttime).seconds)
  da = DaVincv()
  da.objects = vg.fringe_stack
  # da.objects = (vg.img_stack)
  da.add_plotter(da.imshow)
  da.show()
