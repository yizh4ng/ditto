import cv2
import numpy as np

from roma import Nomear
from ditto.image.image import DigitalImage


class FringeEncoder(DigitalImage):

  def __init__(self, img: np.ndarray, radius_range=(80, 81), uncentral_range=((120,121), (120, 121)), **kwargs):
    super(FringeEncoder, self).__init__(img, **kwargs)
    # self.radius = radius
    # self.uncentral = uncentral
    self.radius_range = radius_range
    self.uncentral_range = uncentral_range
  # region: Properties

  @property
  def F(self) -> np.ndarray:
    return np.fft.fftshift((np.fft.fft2(self.to_phase)))
    # return self.get_from_pocket(
    #   'fft2(x)', initializer=lambda: np.fft.fftshift((np.fft.fft2(self.to_phase))))

  @property
  def mask(self) -> np.ndarray:
    # TODO: consider apodized band-pass filter
    def _get_mask():
      H, W = self.Sc.shape
      X, Y = np.ogrid[:H, :W]
      return np.sqrt((X - H/2) ** 2 + (Y - W/2) ** 2) <= np.random.randint(*self.radius_range)
    return _get_mask()
    # return self.get_from_pocket('mask_of_+1_point', initializer=_get_mask)

  @property
  def to_phase(self):
    return np.exp(1j * self.img)

  @property
  def masked(self):
    masked = self.mask * self.F
    return masked

  @property
  def uncentralized_signal(self) -> np.ndarray:
    def _uncentralized_signal():
      masked = (self.F * self.mask)
      CI, CJ = [s % 2 for s in self.Fc.shape]
      # pi, pj = self.uncentral[0], self.uncentral[1]
      l = np.random.randint(*self.uncentral_range[0])
      theta = (np.random.randint(*self.uncentral_range[1]) / 360) * 2 * np.pi
      rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
      pos = np.dot(rotation_matrix, (l, 0))
      pi, pj = int(pos[0]), int(pos[1])
      pos = np.roll(masked.copy(), shift=(+ pi,+ pj), axis=(0, 1))
      neg = np.conjugate(np.rot90(pos, 2))
      neg = np.roll(neg, shift=((CI+1) % 2, (CJ+1) % 2), axis=(0, 1))
      return pos + neg
    return _uncentralized_signal()
    # return self.get_from_pocket('homing_masked_Fc', initializer=_uncentralized_signal)

  @property
  def extracted_fringe(self) -> np.ndarray:
    return np.real(np.fft.ifft2(np.fft.ifftshift(self.uncentralized_signal)))
    # return self.get_from_pocket(
    #   'extracted_image',
    #   initializer=lambda: np.real(np.fft.ifft2(np.fft.ifftshift(self.uncentralized_signal))))


  def save(self, path):
    cv2.imwrite(path, self.extracted_fringe)
