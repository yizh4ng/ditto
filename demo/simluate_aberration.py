import cv2
from lambo.data_obj.interferogram import Interferogram
import numpy as np
from ditto.aberration_generator.quadratic_aberration_generator import QuadraticAberrationGenerator
from ditto.aberration_generator.kaiqiang_generator import KaiqiangAberrationGenerator
from lambo.gui.vinci.vinci import DaVinci
from skimage.restoration import unwrap_phase


'''

Read Interferograms and background

'''


# fringe = cv2.imread('G:/projects/data_old/05-bead/100007.tif')[:, :, 0]
# bg = cv2.imread('G:/projects/data_old/05-bead/100008.tif')[:, :, 0]
fringe = cv2.imread('G:/projects/data_old/05-bead/100011.tif')[:, :, 0]
bg = cv2.imread('G:/projects/data_old/05-bead/100012.tif')[:, :, 0]

'''

Retrieve phase 

'''
radius = 80
ifg = Interferogram(fringe, radius=radius, bg_array=bg)
ifg.booster=True
ifg._backgrounds[0].booster = True
ifg_fft = ifg.extracted_image
ifg_fft_intensity = np.abs(ifg_fft)
ifg_fft_real = np.real(ifg_fft) / ifg_fft_intensity # We should not condsider the Intensity inforamation, because we cannot simluate intensity information of the simulated aberration
ifg_fft_imag = np.imag(ifg_fft) / ifg_fft_intensity
ifg_phase = ifg.flattened_phase


'''

Simulate aberration

'''

config =  {'random_rotate': True,
           'long_axis_range': (1, 1.25),
           'short_axis_range': (0.75, 1),
           'height_range': [30, 50],
           'offset_range': [-50, -40],
           'radius_range': [5, 10]}

ifg_phase_with_simulated_aber = QuadraticAberrationGenerator(config).generate(ifg_phase)

# config = {'grid_num_range': (3, 5),
#           'height_range': (30, 50)}
#
# ifg_phase_with_simulated_aber = KaiqiangAberrationGenerator(config).generate(ifg_phase)

'''

Retrieve Real and Imaginary part

'''


ifg_fft_new = np.exp(1j * ifg_phase_with_simulated_aber)
ifg_fft_real_new = np.real(ifg_fft_new)
ifg_fft_imag_new = np.imag(ifg_fft_new)

'''

Visualize

'''
da = DaVinci()
da.objects = [fringe,
<<<<<<< Updated upstream
              ifg.extracted_angle_unwrapped,
              # ifg_phase,
              # ifg.extracted_angle,
              # ifg_fft_imag, ifg_fft_real,
              # np.real(ifg_fft), ifg_fft_real, ifg_fft_imag, unwrap_phase(np.angle(ifg_fft_real + 1j * ifg_fft_imag)),
              np.real(ifg_fft), np.imag(ifg_fft), ifg_fft_real, ifg_fft_imag,
              ifg_fft_real_new, ifg_fft_imag_new,
              # ifg_fft_real_new * ifg_fft_imag_new,
              # ifg_phase_with_simulated_aber,
              unwrap_phase(np.angle(ifg_fft_new))
=======
              # ifg.extracted_angle_unwrapped, ifg_phase,
              # ifg.extracted_angle,
              # ifg_fft_imag, ifg_fft_real,
              # np.real(ifg_fft), ifg_fft_real, ifg_fft_imag, unwrap_phase(np.angle(ifg_fft_real + 1j * ifg_fft_imag)),
              ifg_fft_real_new, ifg_fft_imag_new, ifg_fft_real_new * ifg_fft_imag_new,
              # ifg_phase_with_simulated_aber,
              # unwrap_phase(np.angle(ifg_fft_new))
>>>>>>> Stashed changes
              ]
da.add_plotter(da.imshow)
da.add_plotter(da.plot3d)
da.show()
