from ditto.image.fringe_simulater import FringeSimulater


import cv2
fringe = cv2.imread('G:/projects/data_old/05-bead/100007.tif')[:, :, 0]

from lambo.gui.vinci.vinci import DaVinci
da = DaVinci()
da.objects = [fringe]

N = 10
for i in range(0 , N):
  current_angle = int(360 / N * i)

  ImageConfig = {}
  # control the radius of the cropped spectrum
  ImageConfig['radius_range'] = [200, 201]
  # control how much to move away the homing +1 spectrum [[distance_low, distance_high], [theta_low, theta_high]
  ImageConfig['uncentral_range'] = [[350, 351], [current_angle, current_angle+1]]

  fringe_simulater = FringeSimulater(fringe, ImageConfig)
  simulated_fringe = fringe_simulater.generate()
  da.objects.append(simulated_fringe)
  fringe_simulater.release()

da.add_plotter(da.imshow_pro)
da.set_clim(0, 255) # more clear to compare original fringes and the simulated one
da.show()
