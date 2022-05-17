from ditto.image.fringe_simulater import FringeSimulater

ImageConfig = {}
# control the radius of the cropped spectrum
ImageConfig['radius_range'] = [200, 201]
# control how much to move away the homing +1 spectrum [[distance_low, distance_high], [theta_low, theta_high]
ImageConfig['uncentral_range'] = [[300, 400], [0, 360]]

import cv2
fringe = cv2.imread('G:/projects/data_old/05-bead/100007.tif')[:, :, 0]
fringe_simulater = FringeSimulater(fringe, ImageConfig)


from lambo.gui.vinci.vinci import DaVinci
da = DaVinci()
da.objects = [fringe, fringe_simulater.generate()]
da.add_plotter(da.imshow_pro)
da.set_clim(0, 255) # more clear to original fringes and the simulated one
da.show()
