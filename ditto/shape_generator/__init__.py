# from .shape import *
# from .square import *

from ditto.shape_generator.shape_generator import Shape_generator
from ditto.shape_generator.polygon_generator import Polygon_generator
from ditto.shape_generator.square_generator import Square_generator
from ditto.shape_generator.ellipse_generator import Ellipse_generator



shape_dict = {'Square': Square_generator,
              'Ellipse': Ellipse_generator,
              'Polygon': Polygon_generator}
