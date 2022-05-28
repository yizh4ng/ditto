import numpy as np
from ditto.image.interferogram import Interferogram
from ditto.aberration_generator import aberration_generator
from roma.ideology.noear import Nomear

def FFT(img):
  return np.fft.fftshift(np.fft.fft2(img))

def mask(ci, cj, radius, shape):
  H, W = shape
  X, Y = np.ogrid[:H, :W]
  return np.sqrt((X - ci) ** 2 + (Y - cj) ** 2) <= radius

def IFFT(img):
  return np.fft.ifft2(np.fft.ifftshift(img))


class FringeSimulater(Nomear):
  def __init__(self, fringe, config):
    super(FringeSimulater, self).__init__()
    self.radius_range = config['radius_range']
    self.radius = np.random.randint(*self.radius_range)
    self.ifg = Interferogram(fringe, radius=self.radius)
    self.uncentral_range = config['uncentral_range']
    # self.aberration_generator = aberration_generator

  def generate_random_peak_index(self):
    l = np.random.randint(*self.uncentral_range[0])
    theta = (np.random.randint(*self.uncentral_range[1]) / 360) * 2 * np.pi
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return np.dot(rotation_matrix, (l, 0))


  def raw_ifg_low_frequency(self):
    def _raw_ifg_low_frequency():
      peaki, peakj = peak_index = self.ifg.peak_index
      # print(peak_index)
      H, W = img_size = self.ifg.img.shape
      peak_mask_1 = mask(peaki, peakj, self.radius, img_size)
      peak_mask_2 = mask(H + (H + 1) % 2 - peaki, W + (W + 1) % 2 - peakj, self.radius,
                         img_size)
      peak_mask = peak_mask_1 + peak_mask_2
      low_fre = np.real(IFFT(~peak_mask * FFT(self.ifg.img)))
      return low_fre
    return self.get_from_pocket('raw_ifg_low_frequency', initializer=_raw_ifg_low_frequency)

  def generate_simulated_high_frequency(self):
    H, W = self.ifg.img.shape
    new_peak = self.generate_random_peak_index().astype(int)
    # new_peak = (new_peak + np.array([H/2, W/2])).astype(int)
    low_fre = self.raw_ifg_low_frequency()
    # if self.aberration_generator is not None:
    #   peak_angle = self.ifg.extracted_angle_unwrapped
    #   peak_angle = self.aberration_generator.generate(peak_angle)
    #   peak = FFT(np.exp(1j * peak_angle))
    # else:
    peak = self.ifg.homing_signal
    peak_uncentralized = np.roll(peak, shift=new_peak, axis=(0, 1))
    peak_dual = np.conjugate(np.rot90(peak_uncentralized, 2))
    peak_dual = np.roll(peak_dual, shift=((H + 1) % 2, (W + 1) % 2),
                          axis=(0, 1))

    simulated_high_fre = np.real(IFFT(peak_uncentralized + peak_dual))
    return simulated_high_fre

  def generate(self):
    simulated_fringe = self.raw_ifg_low_frequency() + self.generate_simulated_high_frequency()
    # control pixel value range
    simulated_fringe[simulated_fringe<0] = 0
    simulated_fringe[simulated_fringe>255] = 255
    return simulated_fringe


if __name__ == '__main__':
  from ditto import ImageConfig
  ImageConfig['radius_range'] = [200, 201]
  ImageConfig['uncentral_range'] = [[300, 400], [0, 360]]
  import cv2
  fringe = cv2.imread('G:/projects/data_old/05-bead/100007.tif')[:, :, 0]
  # from ditto.aberration_generator.quadratic_aberration_generator import QuadraticAberrationGenerator
  # fringe_simulater = FringeSimulater(fringe, ImageConfig, QuadraticAberrationGenerator(ImageConfig))
  fringe_simulater = FringeSimulater(fringe, ImageConfig)


  from lambo.gui.vinci.vinci import DaVinci
  da = DaVinci()
  da.objects = [fringe, fringe_simulater.generate()]
  da.add_plotter(da.imshow_pro)
  ig = Interferogram(fringe_simulater.generate(), radius=80)
  ig.dashow()
  da.show()
