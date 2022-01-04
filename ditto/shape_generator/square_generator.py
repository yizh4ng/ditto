from ditto.shape_generator import Polygon_generator
import numpy as np

class Square_generator(Polygon_generator):
  def __init__(self, **kwargs):
    super(Square_generator, self).__init__(**kwargs)
    self.num_point_range = (4, 5)
