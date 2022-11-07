import numpy as np
from ditto.shape_generator.shape.zernike import Zernike
from ditto.shape_generator import BaseGenerator



class ZernikeGenerator(BaseGenerator):
  _name = 'Zernike'

  def __init__(self, zernike_coefficient: list):
    super(ZernikeGenerator, self).__init__()
    self.zernike_coefficient = np.random.uniform(low=-1.1, high=-1, size=37) * np.array([0.5 ** i for i in range(37)])
    self.num = 1


  def generate_one(self, img):
    return Zernike(self.zernike_coefficient, (int(0.5 * img.shape[0]),
                                              int(0.5 * img.shape[1])))


  def generate(self, img):
    shapes = []
    for _ in range(self.num):
      shapes.append(self.generate_one(img))

    return shapes





ZernikeGenerator.register()
