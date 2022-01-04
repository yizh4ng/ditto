from ditto import Painter
from lambo.gui.vinci.vinci import DaVinci
import numpy as np

from ditto.image.config import Config as ImageConfig
from ditto.video.config import Config as VideoConfig
from scipy.signal import convolve2d

class Video_generator():
  def __init__(self, video_length, move_range, rotate_range):
    self.painter = Painter(**ImageConfig)
    self.img_config = ImageConfig
    self.img_size = self.painter.img.shape
    self.video_length = video_length
    self.painter.paint_samples()
    self.first_frame = self.painter.img
    self.img_stack = [self.first_frame]
    self.fringe_stack = [self.painter.extracted_fringe]
    self.shapes = self.painter.shapes
    self.move_range = move_range
    self.rotate_range = rotate_range

  def step(self):
    W, H = self.img_size[0], self.img_size[1]
    img = np.zeros((W, H))
    for s in self.shapes:

      s.move(offset=(np.random.randint(*self.move_range[0]),
                     np.random.randint(*self.move_range[1])))
      s.rotate(angle=np.random.randint(*self.rotate_range))
      img = s.draw(img)
    if self.img_config['smooth']:

      conv_size = 2 * self.img_config['smooth'] + 1
      img = convolve2d(img,
                          (np.zeros([conv_size, conv_size]) + 1 / (
                              conv_size ** 2)),
                          boundary='symm',
                          mode='same')
    # self.painter.img = img[W:-W, H:-H]
    self.painter.img = img
    self.img_stack.append(self.painter.img)
    self.fringe_stack.append(self.painter.extracted_fringe)

  def generate(self):
    for _ in range(self.video_length - 1):
      self.step()

if __name__ == '__main__':
  vg = Video_generator(**VideoConfig)

  vg.generate()

  da = DaVinci()
  da.objects = vg.fringe_stack
  # da.objects = vg.img_stack
  da.add_plotter(da.imshow)
  da.show()
