from ditto.aberration_generator.tilt_abberation_generator import TiltAbberationGenerator
from ditto.aberration_generator.quadratic_aberration_generator import QuadraticAberrationGenerator
from ditto.aberration_generator.kaiqiang_generator import KaiqiangAberrationGenerator

abberation_dict = {'tilt_abberation': TiltAbberationGenerator,
                   'quadratic_aberration': QuadraticAberrationGenerator,
                   'kaiqiang_aberration': KaiqiangAberrationGenerator}