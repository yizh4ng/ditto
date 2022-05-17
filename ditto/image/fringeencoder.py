import cv2
import numpy as np

from ditto.image.interferogram import Interferogram
from roma import Nomear
from ditto.image.image import DigitalImage


class FringeEncoder(DigitalImage):

  def __init__(self, img: np.ndarray, config, abberation_image=None, **kwargs):
    super(FringeEncoder, self).__init__(img, **kwargs)
    # self.radius = radius
    # self.uncentral = uncentral
    self.config = config
    self.radius_range = config['radius_range']
    self.uncentral_range = config['uncentral_range']
    self.abberation_image = abberation_image
    self.bg_ig = None
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

  @property
  def physics_based_fringe(self):
    W, H = self.img.shape[0], self.img.shape[1]
    img = np.expand_dims(self.img, axis=-1)
    i_coords, j_coords = np.meshgrid(range(W), range(H), indexing='ij')
    i_coords = np.expand_dims(i_coords,axis=-1)
    j_coords = np.expand_dims(j_coords,axis=-1)
    img_3d = np.concatenate((i_coords, j_coords, img), axis=-1)

    source_light_pos = self.config['source_light_pos']
    # ref_source_pos = self.config['ref_light_pos']
    ref_offset = self.config['ref_light_offset']
    ref_angle = self.config['ref_light_angle']
    if isinstance(ref_angle, tuple):
      ref_angle = np.random.uniform(*ref_angle)
    ref_height = self.config['ref_light_height']
    ref_source_pos = (ref_offset* np.cos(ref_angle),
                      ref_offset* np.sin(ref_angle), ref_height)

    distance = img_3d - np.array(source_light_pos)
    distance_ref = img_3d - np.array(ref_source_pos)

    distance_difference = distance + distance_ref
    phase_difference = np.sqrt(np.sum(np.square(distance_difference), axis=-1))
    phase_difference_ = np.copy(phase_difference)

    if self.config['frequency'][0] == self.config['frequency'][1]:
      frequency = self.config['frequency'][0]
    else:
      frequency = np.random.uniform(self.config['frequency'][0],
                                    self.config['frequency'][1])

    if self.config['first_phase'][0] == self.config['first_phase'][1]:
      first_phase = self.config['first_phase'][0]
    else:
      first_phase = np.random.uniform(self.config['first_phase'][0],
                                      self.config['first_phase'][1])

    if self.abberation_image is not None:
      self.bg_ig = Interferogram(self.abberation_image,
                                 radius=200)
      phase_difference += self.bg_ig.extracted_angle_unwrapped

      phase_map = np.cos(frequency * phase_difference + first_phase)
      if self.config['with_abberation']:
        abberation = self.bg_ig.low_frequency_filter
        # light_attenuation = 1/(1 + 0.000001 * phase_difference_)
        high_frequency = self.bg_ig.high_frequency_filter
        beta = np.max(np.real(self.bg_ig.high_frequency_filter))
        index = np.unravel_index(np.argmax(high_frequency),
                                 high_frequency.shape)
        alpha = abberation[index]

        offset = (alpha - np.sqrt(alpha ** 2 - beta ** 2)) / 2
        abberation_correct = abberation-offset
        abberation_correct[abberation_correct<0] = 0
        phase_map =   abberation + 2 * np.sqrt((abberation_correct) * offset) *  phase_map
      self.bg_ig.release()
    else:
      phase_map = np.cos(frequency * phase_difference + first_phase)

    # phase_difference += np.random.normal(0, 0.01, (W, H))
    if self.config['normalize']:
      phase_map -= np.min(phase_map)
      phase_map = phase_map / np.max(phase_map)
    if self.config['binary']:
      mean = np.mean(phase_map)
      zero_indices = phase_map <= mean
      one_indices = phase_map > mean
      phase_map[zero_indices] = 0
      phase_map[one_indices] = 1
    return phase_map


if __name__ == '__main__':
  Config = {'img_size': (512, 512),
            'radius_range': (50, 51),
            'uncentral_range': ((60, 61), (60, 61)),
            'source_light_pos': (0, 0, 100),
            'ref_light_pos': (100, 100, 50),
            'suffle': True,
            'smooth': 2,
            'draw_over': False,
            # 'tilt_abberation': {'slop_direction':(0,1),
            #                     'slop':0.01,
            #                     'lowest_height': 0},
            # 'Square':{'radius_range': (0.3, 0.3),
            #             'num_range': (1, 2),
            #             'num_point_range':(4, 6),
            #             'height_range': (0, 1),
            #             'random_rotate': True},
            'Ellipse': {'radius_range': (0.1, 0.2),
                        'long_axis_range': (1.0, 1.1),
                        'short_axis_range': (0.9, 1.0),
                        'num_range': (4, 5),
                        'height_range': (5, 10),
                        'random_rotate': True
                        },
            # 'Polygon': {'radius_range': (0.2, 0.2),
            #             'num_range': (4, 5),
            #             'num_point_range':(3, 4),
            #             'height_range': (5, 10),
            #             'random_rotate': True,
            #             'irregular': False},
            }
  fe= FringeEncoder(np.array([[1,2], [3,4]]), config=Config)
  fe = fe.physics_based_fringe

