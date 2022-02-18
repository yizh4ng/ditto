import numpy as np
import matplotlib.pyplot as plt
import roma

from collections import OrderedDict
from typing import Optional, Union, List, Tuple
from lambo.data_obj.image import DigitalImage
from lambo.gui.pyplt import imshow
from lambo.gui.pyplt.events import bind_quick_close
from lambo.gui.vinci.vinci import DaVinci
from lambo.maths.geometry.misc import rotate_coord
from lambo.maths.geometry.plane import fit_plane_adaptively
from lambo.misc.timer import Timer

from scipy import ndimage
from skimage.restoration import unwrap_phase


class Interferogram(DigitalImage):

  class Keys(object):
    fourier_prior = '_fourier_prior_'
    file_path = '_file_path_'

  def __init__(self, img, bg_array=None, radius=None, lambda_0=None,
               delta_n=None, peak_index=None, **kwargs):
    # Call parent's initializer
    super(Interferogram, self).__init__(img, **kwargs)

    # Attributes
    self.radius = roma.check_type(radius, int, nullable=True)
    self.lambda_0 = roma.check_type(lambda_0, float, nullable=True)
    self.delta_n = roma.check_type(delta_n, float, nullable=True)
    self._backgrounds: Optional[List[Interferogram]] = None

    # Set peak index if provided
    self._peak_index: Optional[Tuple[int, int]] = None
    if peak_index is not None: self.peak_index = peak_index

    # Set background if provided
    if bg_array is not None: self.set_background(bg_array)

    # Other field (should be cloned)
    self.sample_token = None
    self.setup_token = None
    self.tag = ''

  # region: Properties

  @property
  def clone(self):
    bg_array = None
    if self.default_background is not None: bg_array = self.bg_array
    ig = Interferogram(self.img, radius=self.radius, peak_index=self.peak_index,
                       bg_array=bg_array)

    # Clone other fields
    ig.sample_token = self.sample_token
    ig.setup_token = self.setup_token
    ig.tag = self.tag
    return ig

  @property
  def fourier_prior(self) -> np.ndarray:
    if not hasattr(self, self.Keys.fourier_prior):
      self._set_fourier_prior(self.peak_index)
    return getattr(self, self.Keys.fourier_prior)

  @property
  def peak_mask(self):
    # Initialize mask
    mask = np.ones_like(self.img)
    h, w = mask.shape
    # Put mask
    ci, cj = h // 2, w // 2
    d = int(0.05 * min(h, w))
    # mask[ci-d:ci+d, :] = 0
    mask[ci-d:ci+d, cj-d:cj+d] = 0
    mask[:, cj+d:] = 0
    return mask

  @property
  def log_Sc(self) -> np.ndarray:
    return np.log(self.Sc + 1)

  @property
  def default_background(self):
    if isinstance(self._backgrounds, list): return self._backgrounds[0]
    return None

  @property
  def bg_array(self) -> np.ndarray:
    return self.default_background.img

  @property
  def peak_index(self) -> Tuple[int, int]:
    """Peak index should be localized since this variable takes no space"""
    def _find_peak() -> Tuple[int, int]:
      region = self.Sc
      index = np.unravel_index(np.argmax(region * self.peak_mask), region.shape)
      self._set_fourier_prior(index)
      return index

    if self._peak_index is None: self._peak_index = _find_peak()
    return self._peak_index

  @peak_index.setter
  def peak_index(self, val):
    assert isinstance(val, tuple) and len(val) == 2
    self._peak_index = roma.check_type(val, inner_type=int)

  @property
  def mask(self) -> np.ndarray:
    # TODO: consider apodized band-pass filter
    def _get_mask():
      H, W = self.Sc.shape
      ci, cj = self.peak_index
      X, Y = np.ogrid[:H, :W]
      return np.sqrt((X - ci)**2 + (Y - cj)**2) <= self.radius
    return self.get_from_pocket('mask_of_+1_point', initializer=_get_mask)

  @property
  def mask_dual(self) -> np.ndarray:
    def _get_mask_():
      CI, CJ = [s % 2 for s in self.Fc.shape]
      mask_dual = np.rot90(self.mask, 2)
      mask_dual = np.roll(mask_dual, shift=((CI + 1) % 2, (CJ + 1) % 2), axis=(0, 1))
      return mask_dual
    return self.get_from_pocket('mask_of_-1_point', initializer=_get_mask_)

  @property
  def high_frequency_filter(self):
    def _hff():
      mask = np.logical_or(self.mask_dual, self.mask)
      masked = self.Fc * mask
      return np.real(np.fft.ifft2(np.fft.ifftshift(masked)))
    return self.get_from_pocket('high_frequency_filter', initializer=_hff)

  @property
  def low_frequency_filter(self):
    def _lff():
      mask = np.logical_or(self.mask, self.mask_dual)
      masked = self.Fc * np.logical_not(mask)
      return np.real(np.fft.ifft2(np.fft.ifftshift(masked)))
    return self.get_from_pocket('low_frequency_filter', initializer=_lff)

  @property
  def homing_signal(self) -> np.ndarray:
    def _homing_signal():
      masked = self.Fc * self.mask
      CI, CJ = [s // 2 for s in self.Fc.shape]
      pi, pj = self.peak_index
      return np.roll(masked, shift=(CI - pi, CJ - pj), axis=(0, 1))
    return self.get_from_pocket('homing_masked_Fc', initializer=_homing_signal)

  @property
  def extracted_image(self) -> np.ndarray:
    return self.get_from_pocket(
      'extracted_image',
      initializer=lambda: np.fft.ifft2(np.fft.ifftshift(self.homing_signal)))

  @property
  def extracted_angle(self) -> np.ndarray:
    return self.get_from_pocket(
      'extracted_angle', initializer=lambda: np.angle(self.extracted_image))

  @property
  def extracted_angle_unwrapped(self) -> np.ndarray:
    return self.get_from_pocket(
      'extracted_angle_unwrapped',
      initializer=lambda: unwrap_phase(self.extracted_angle))

  @property
  def corrected_image(self) -> np.ndarray:
    assert all([isinstance(self._backgrounds[0], Interferogram),
                self.size == self._backgrounds[0].size])
    return self.get_from_pocket('corrected_image', initializer=lambda: (
        self.extracted_image / self._backgrounds[0].extracted_image))

  @property
  def corrected_intensity(self) -> np.ndarray:
    assert all([isinstance(self._backgrounds[0], Interferogram),
                self.size == self._backgrounds[0].size])
    return self.get_from_pocket(
      'corrected_intensity',
      initializer=lambda: np.log(np.abs(self.corrected_image)))

  @property
  def corrected_phase(self) -> np.ndarray:
    """Phase information after subtracting background"""
    assert all([isinstance(self._backgrounds[0], Interferogram),
                self.size == self._backgrounds[0].size])
    return self.get_from_pocket(
      'retrieved_phase', initializer=lambda: np.angle(self.corrected_image))

  @property
  def derivative_1(self) -> np.ndarray:
    def _calculate_grad1():
      x = ndimage.sobel(self.extracted_angle_unwrapped, axis=0, mode='constant')
      y = ndimage.sobel(self.extracted_angle_unwrapped, axis=1, mode='constant')
      return np.stack([x, y], axis=-1)
    return self.get_from_pocket('derivative_1', initializer=_calculate_grad1)

  @property
  def derivative_1_amp(self) -> np.ndarray:
    def _calculate_grad1_amp():
      return np.sqrt(np.sum(self.derivative_1 * self.derivative_1, axis=-1))
    return self.get_from_pocket(
      'derivative_1_amp', initializer=_calculate_grad1_amp)

  @property
  def aberration(self) -> np.ndarray:
    return self.get_from_pocket('aberration', initializer=lambda: (
        self.extracted_angle_unwrapped - self.flattened_phase))

  @property
  def unwrapped_phase(self) -> np.ndarray:
    """Result after performing phase unwrapping"""
    return self.get_from_pocket(
      'unwrapped_phase', initializer=lambda: unwrap_phase(self.corrected_phase))

  @property
  def bg_plane_info(self) -> np.ndarray:
    """(bg, p, r)"""
    _fit_plane = lambda: fit_plane_adaptively(
      self.unwrapped_phase.copy(), **self.flatten_configs)
    return self.get_from_pocket('bg_plane_info', initializer=_fit_plane)

  @property
  def bg_plane(self) -> np.ndarray:
    return self.bg_plane_info[0]

  @property
  def bg_flatness(self):
    return self.bg_plane_info[2]

  @property
  def bg_slope(self):
    a, b, c = self.bg_plane_info[1]
    return np.sqrt(a**2 + b**2) * 1000

  @property
  def flatten_configs(self) -> dict:
    return self.get_from_pocket('flatten_configs', default={})

  @property
  def flattened_phase(self) -> np.ndarray:
    def _flattened_phase():
      phase = self.unwrapped_phase.copy()
      bg_plane = self.bg_plane_info[0]
      flattened = phase - bg_plane
      flattened = np.maximum(flattened, 0)
      return flattened
    return self.get_from_pocket('flattened_phase', initializer=_flattened_phase)

  @property
  def sample_height(self):
    def _height():
      lambda_0 = roma.check_type(self.lambda_0, float)
      delta_n = roma.check_type(self.delta_n, float)
      return self.flattened_phase * lambda_0 / (2 * np.pi * delta_n)
    return self.get_from_pocket('sample_height', initializer=lambda: _height)

  # endregion: Properties

  # region: Private Methods

  def _set_fourier_prior(self, peak_index: Tuple[int, int]):
    H, W = self.size
    ci, cj = H // 2, W // 2
    pi, pj = peak_index
    i, j = pi - ci, pj - cj
    setattr(self, self.Keys.fourier_prior, np.array([i / H, j / W]))

  def _get_unit_vector(self, v: Optional[np.ndarray] = None):
    """Used to get the (omaga=0, r=1.0) vector for generating
       fourier basis stack"""
    assert self.radius > 0
    if v is None: v = self.fourier_prior
    d = np.linalg.norm([
      pi - L // 2 for L, pi in zip(self.size, self.peak_index)])
    return v / d * self.radius

  # endregion: Private Methods

  # region: Public Methods

  def get_model_input(self, index):
    if index == 1: return self.img
    elif index == 2: return np.stack(
      [np.real(self.extracted_image), np.imag(self.extracted_image)], axis=-1)
    elif index == 3: return np.stack(
      [np.abs(self.extracted_image), self.extracted_angle], axis=-1)
    elif index == 9: return self.extracted_angle_unwrapped
    else: raise KeyError('!! index must be in (1, 2, 3)')

  def get_fourier_prior(self, L, angle=0, r=1.0, fmt='default',
                        rotundity=False):
    # Get location of +1 point
    loc = self.fourier_prior
    if angle != 0: loc = rotate_coord(loc, angle)
    if r != 1.0: loc = [r * l for l in loc]
    return self.get_fourier_basis(*loc, L, fmt=fmt, rotundity=rotundity)

  def get_fourier_dual_basis(
      self, L=None, omega=30, rs=(0.3, 0.6, 0.9), return_dict=False, rotundity=True,
      angle=0, max_angle=360, peak_and_uv=False, include_origin=False):
    # TODO: BETA
    if peak_and_uv:
      return np.stack([self.fourier_prior, self._get_unit_vector()])

    # Init an empty dict to fill
    base_dict = OrderedDict()

    def gather(coord, a, r):
      assert len(coord) == 2
      comp = self.get_fourier_basis(coord[0], coord[1], L=L, fmt='comp',
                                    rotundity=rotundity)
      base_dict[(a, r)] = comp

    # Initialize with center component
    loc = self.fourier_prior
    if angle != 0: loc = rotate_coord(loc, angle)

    if include_origin: gather(loc, 0, 0)

    # Rotate loc around +1 point and gather corresponding basis
    uv = self._get_unit_vector(loc)
    for a in range(0, max_angle, omega):
      _uv = rotate_coord(uv, a)
      for r in rs:
        assert 0 < r <= 1
        gather(loc + r * _uv, a, r)

    # Return accordingly
    if return_dict: return base_dict
    return list(base_dict.values())

  def get_fourier_prior_stack(
      self, L, angle, omega=30, rs=(0.3, 0.6, 0.9), fmt='real'):
    loc = self.fourier_prior
    if angle != 0: loc = rotate_coord(loc, angle)

    # Initialize with center component TODO: --
    # priors = [self.get_fourier_basis(*loc, L, fmt=fmt)]
    priors = []

    # Rotate loc with around +1 point and append corresponding basis
    # uv = self._get_unit_vector(loc) TODO: --
    for a in range(0, 180, omega):
      _uv = rotate_coord(loc, a)
      for r in rs:
        assert r > 0
        # _loc = loc + r * _uv #TODO: --
        _loc = r * _uv
        priors.append(self.get_fourier_basis(*_loc, L, fmt=fmt, rotundity=True))

    # Stack/Concatenate priors and return
    if len(priors[0].shape) == 2: return np.stack(priors, axis=-1)
    assert len(priors[0].shape) == 3
    return np.concatenate(priors, axis=-1)

  def rotate(self, angle: float):
    img = self.rotate_image(self.img, angle)
    img = self.get_downtown_area(img)
    bg = self.rotate_image(self.bg_array, angle)
    bg = self.get_downtown_area(bg)

    radius = int(self.radius / min(self.size) * img.shape[0])
    ig = Interferogram(img, bg, radius)
    return ig

  def soft_mask(self, alpha=0.1, mask_min=0.1):
    x = self.flattened_phase
    threshold = alpha * np.max(x)
    mask = x > threshold
    soft_mask = 1.0 * mask + (1.0 - mask) * x / threshold + mask_min
    return soft_mask

  def set_background(self, bg: Union[np.ndarray, List[np.ndarray]]):
    if self.radius is None:
      raise ValueError('!! radius must be specified before setting background')

    if isinstance(bg, np.ndarray): bg = [bg]
    roma.check_type(bg, list, inner_type=np.ndarray)
    bgig_list = [Interferogram(b, radius=self.radius) for b in bg]

    for bg_ig in bgig_list:
      assert bg_ig.size == self.size
      bg_ig.peak_index = self.peak_index

    self._backgrounds = bgig_list

  @classmethod
  def imread(cls, path, bg_path=None, radius=None, peak_index=None, **kwargs):
    img = super().imread(path, return_array=True)
    bg = super().imread(bg_path, return_array=True) if bg_path else None
    ig = Interferogram(img, bg, radius, peak_index=peak_index)
    # Set filename to interferogram
    ig.put_into_pocket(cls.Keys.file_path, path, local=True)
    if bg_path is not None:
      ig.default_background.put_into_pocket(
        cls.Keys.file_path, bg_path, local=True)
    return ig

  def dashow(self, size=7, show_calibration=True, show_grad1=False):
    if self._backgrounds is None: show_calibration = False

    da = DaVinci('Interferogram Analyzer', size, size)
    da.add_imshow_plotter(self.img, 'Interferogram With Sample')
    if show_calibration:
      da.add_imshow_plotter(self.bg_array, 'Interferogram for Calibration')

    da.add_imshow_plotter(
      np.log(self.Sc + 1), 'Centered Spectrum (log)',
      lambda: plt.gca().add_artist(plt.Circle(
        list(reversed(self.peak_index)), self.radius, color='r', fill=False)))
    # da.add_imshow_plotter(
    #   np.real(self.extracted_image), 'Extracted Image (Real)')
    # da.add_imshow_plotter(
    #   np.imag(self.extracted_image), 'Extracted Image (Imag)')
    da.add_imshow_plotter(
      np.abs(self.extracted_image), 'Extracted Image (Magnitude)')
    da.add_imshow_plotter(self.extracted_angle, 'Extracted Angle')
    da.add_imshow_plotter(
      self.extracted_angle_unwrapped, 'Extracted Angle (unwrapped)')

    if show_grad1:
      # da.add_plotter(lambda ax: da.histogram(self.derivative_1_amp, ax))

      grad1 = self.derivative_1_amp
      grad1 = np.log(grad1 + 1)
      # grad1 = np.minimum(grad1, 25)
      da.add_imshow_plotter(grad1, '|1st Order Gradient|')
      da.add_imshow_plotter(self.flattened_phase, 'Flattened Phase')

    if show_calibration:
      da.add_imshow_plotter(
        self._backgrounds[0].extracted_angle, 'Background Angle')
      da.add_imshow_plotter(
        self._backgrounds[0].extracted_angle_unwrapped,
        'Background Angle (unwrapped)')
      # da.add_imshow_plotter(self.corrected_intensity, 'Corrected Intensity')
      da.add_imshow_plotter(self.corrected_phase, 'Calibrated Phase')
      da.add_imshow_plotter(self.unwrapped_phase, 'Unwrapped Phase')
      roma.console.show_status('Fitting background plane ...')
      da.add_imshow_plotter(self.flattened_phase, 'Flattened Phase')

      def _plot_3d(ax3d):
        da.plot3d(self.flattened_phase, ax3d, title='Flattened Phase (3D)')
      da.add_plotter(_plot_3d)

    da.show()

  def set_flatten_configs(self, **configs):
    self.put_into_pocket('flatten_configs', configs)

  def dual_conv(self, kernel: np.ndarray, mu=0, sigma=1) -> np.ndarray:
    from scipy import signal

    if mu is None: mu =  np.mean(self.img)
    im = (self.img - mu) / sigma

    # Sanity check
    assert np.max(np.abs(kernel)) <= 1.0

    # Produce dual kernel
    dual = np.sqrt(1 - np.square(kernel))

    y1 = signal.convolve2d(im, kernel)
    y2 = signal.convolve2d(im, dual)
    y = np.sqrt(np.square(y1), np.square(y2))

    return y

  # endregion: Public Methods

  # region: Analysis

  def phase_histogram(self, unwrapped=True, flattened=True):
    buffer = []
    def _plot_hist(array: np.ndarray, title: str, total: int):
      if total > 1: plt.subplot(1, total, len(buffer) + 1)
      plt.hist(np.ravel(array))
      plt.title('{}, Range: [{:.2f}, {:.2f}]'.format(
        title, np.min(array), np.max(array)))
      buffer.append(title)

    # Plot hist
    total = unwrapped + flattened
    size = 5
    plt.figure(figsize=(size * total, size))
    if unwrapped: _plot_hist(self.unwrapped_phase, 'Unwrapped Phase', total)
    if flattened: _plot_hist(self.flattened_phase, 'Flattened Phase', total)

    # Finalize
    plt.tight_layout()
    bind_quick_close()
    plt.show()

  def flatten_analysis(self, plot3d_in_2=False, show_plane=True, **configs):
    from matplotlib.gridspec import GridSpec
    from matplotlib import cm

    def _plot_img(ax: plt.Axes, array: np.ndarray, title=None, plot3d=False):
      if plot3d:
        H, W = array.shape
        X, Y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        ax.plot_surface(X, Y, array, cmap=cm.coolwarm)
      else:
        ax.imshow(array)
        ax.set_axis_off()
      if title: ax.set_title(title)

    def _plot_hist(array: np.ndarray, title: str):
      plt.hist(np.ravel(array), bins=20, density=True)
      plt.title('{}, Range: [{:.2f}, {:.2f}]'.format(
        title, np.min(array), np.max(array)))

    self.set_flatten_configs(**configs)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    spec = GridSpec(nrows=2, ncols=2, height_ratios=[3, 1])

    left_array = self.unwrapped_phase
    ax = fig.add_subplot(spec[0], projection='3d')
    assert isinstance(ax, plt.Axes)
    _plot_img(ax, left_array, 'Unwrapped Phase', plot3d=True)
    fig.add_subplot(spec[2])
    _plot_hist(left_array, 'Unwrapped Phase')

    # Plot plane if required
    if show_plane:
      H, W = self.bg_plane.shape
      X, Y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
      ax.plot_wireframe(X, Y, self.bg_plane, color='green')
      title = 'Unwrapped Phase, r = {:.1f}'.format(self.bg_flatness)
      title += ', s = {:.1f}'.format(self.bg_slope)
      title += ', b = {:.1f}'.format(self.bg_plane_info[-1])
      ax.set_title(title)

    # Plot right part
    z_lim = ax.get_zlim()

    right_array = self.flattened_phase
    kwargs = {'projection': '3d'}  if plot3d_in_2 else {}
    ax = fig.add_subplot(spec[1], **kwargs)
    _plot_img(ax, right_array, 'Flattened Phase', plot3d=plot3d_in_2)
    fig.add_subplot(spec[3])
    _plot_hist(right_array, 'Flattened Phase')

    # Finalize
    bind_quick_close()
    plt.tight_layout()
    plt.show()

  def analyze_windows(self, N, ignore_gap=0, max_cols=3, show_grid=0):
    """Divide img evenly into N by N parts and do phase retrieval individually.
    The difference between this result and the common non-dividing phase
    retrieval result will be shown.

    :param N: number of each dimension dividing into """
    h, w = [l // N for l in self.size]
    ground_truth =self.unwrapped_phase[:h*N, :w*N]
    assembled = np.zeros_like(ground_truth)
    for i, j in [(i, j) for i in range(N) for j in range(N)]:
      im = self.img[i*h:(i+1)*h, j*w:(j+1)*w]
      bg = self.bg_array[i*h:(i+1)*h, j*w:(j+1)*w]
      ig = Interferogram(im, bg, self.radius // N)
      assembled[i*h:(i+1)*h, j*w:(j+1)*w] = ig.unwrapped_phase

    # Compare
    delta = np.abs(ground_truth - assembled)

    # Ignore gap if required
    if ignore_gap:
      r = ignore_gap
      for i in range(1, N): delta[i * h - r:i * h + r, :] = 0
      for j in range(1, N): delta[:, j * w - r:j * w + r] = 0

    # Show grid if necessary
    if show_grid:
      r = show_grid
      v = np.min(assembled)
      for i in range(1, N): assembled[i * h - r:i * h + r, :] = v
      for j in range(1, N): assembled[:, j * w - r:j * w + r] = v

    # Calculate MSE and plot
    mse = (delta ** 2).mean()
    imshow(ground_truth, assembled, delta, titles=[
      'Ground Truth', 'Assembled {0}x{0}'.format(N),
      'Difference, MSE = {0:.3f}'.format(mse)], max_cols=max_cols)

  def analyze_rotation(self, key='flattened_phase'):
    # Plot non-rotated image on top left
    images, titles = [getattr(self, key)], ['0 Degree']
    # Rotate, retrieve, and plot
    for k in range(1, 4):
      ig = self.rotate(k)
      images.append(getattr(ig, key))
      titles.append('{} Degree'.format(90 * k))

    # Show images
    imshow(*images, titles=titles, max_cols=2)

  def show_fourier_basis(self, L=50, angles=0, rs=1.0, include_imag=False):
    if isinstance(angles, int): angles = (angles,)
    if isinstance(rs, (int, float)): rs = (rs,)

    da = DaVinci('Fourier Basis', height=7, width=7)
    for a in angles:
      im = self.img
      im = self.get_downtown_area(self.rotate_image(im, a))
      da.objects.append(im[:L, :L])
      for r in rs:
        da.objects.append(self.get_fourier_prior(L, a, r, fmt='real'))
        if include_imag: da.objects.append(
          self.get_fourier_prior(L, a, r, fmt='image'))
    da.add_plotter(da.imshow)
    da.show()

  def show_dettol(self, L, angle=0, omega=30, rs=(0.3, 0.6, 0.9),
                  show_what='sc'):
    # Initialize da
    da = DaVinci('Dettol', height=7, width=7, init_as_image_viewer=True)

    # Get basis
    stack = self.get_fourier_prior_stack(L, angle, omega, rs)
    assert len(stack.shape) == 3
    basis = [stack[:, :, i] for i in range(stack.shape[-1])]

    # Show dettol
    if show_what in ('sc', 'spectrum'):
      da.add_image(
        np.sum([DigitalImage(b).Sc for b in basis], axis=0), 'Spectrum')
      da.add_image(
        DigitalImage(self.img[:L, :L]).Sc, 'Spectrum of Interferogram')
      da.show()
      return

    assert show_what in ('basis', 'img', 'im')
    da.objects = basis
    da.show()

  def analyze_dual_conv(self, L, omega=30, rs=(0.3, 0.6, 0.9), merge=False,
                        rotundity=True, show_kernel=True, preprocess=True):
    from roma import console

    base_dict = self.get_fourier_dual_basis(
      L, omega, rs, return_dict=True, rotundity=rotundity)

    da = DaVinci('Dual Conv Analysis', init_as_image_viewer=True)

    # Add all
    if merge: base_dict[(-1, -1)] = np.sum(
      [np.real(b) for _, b in base_dict.items()], axis=0)

    if not show_kernel: da.add_image(self.img, 'Raw Interferogram')

    total = len(base_dict)
    for i, ((a, r), b) in enumerate(base_dict.items()):
      console.print_progress(i, total)

      knl = np.real(b)
      if show_kernel: im = np.real(knl)
      else: im = self.dual_conv(knl, mu=None if preprocess else 0)
      da.add_image(im, 'a={}, r={}'.format(a, r))

    if not show_kernel: da.add_image(self.flattened_phase, 'Flattened Phase')

    console.show_status('Images set to Da Vinci.')
    da.toggle_log()
    da.show()

  def analyze_time(self):
    tm = Timer()
    with tm.tic_toc('Fourier Transform'):
      _ = self.Fc
      _ = self._backgrounds[0].Fc
    with tm.tic_toc('First Order Extraction'):
      _ = self.extracted_image
      _ = self._backgrounds[0].extracted_image
    with tm.tic_toc('Aberration Correction'):
      _ = self.corrected_phase
    with tm.tic_toc('Phase Unwrapping'):
      _ = self.unwrapped_phase
    tm.report()

  # endregion: Analysis


if __name__ == '__main__':
  from pr.pr_agent import PRAgent

  data_dir = r'E:\lambai\01-PR\data'

  trial_id = 1
  sample_id = 2
  # pattern = '*62-*'
  pattern = None
  ig = PRAgent.read_interferogram(
    data_dir, trial_id, sample_id, pattern=pattern, radius=80)
  # ig = ig.rotate(10)

  # ig = Interferogram.imread(r'E:\lambai\01-PR\data\63-Nie system\1.tif',
  #                           radius=70)

  ig.dashow(show_calibration=False, show_grad1=True)
  # ig.analyze_time()
  # ig.analyze_windows(4)
  # ig.show_fourier_basis(21, rs=(1.0,), angles=(5, 10))
  # ig.analyze_dual_conv(17, omega=45, rs=(0.4,), rotundity=True,
  #                      show_kernel=False, preprocess=True)

  # rs = [1]
  # ig.show_dettol(11, angle=0, omega=15, rs=rs, show_what='im')





