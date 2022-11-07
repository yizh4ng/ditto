from ditto.shape_generator import PolygonGenerator
import numpy as np

class SquareGenerator(PolygonGenerator):
  _name = 'Square'

  def __init__(self, **kwargs):
    super(SquareGenerator, self).__init__(**kwargs)
    self.num_point_range = (4, 5)

SquareGenerator.register()
