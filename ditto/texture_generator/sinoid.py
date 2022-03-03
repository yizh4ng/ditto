from ditto.texture_generator.texture_generator import TextureGenerator
from ditto.utili import parse, rotation_matrix
import numpy as np



class SinoidGenerator(TextureGenerator):
  def __init__(self, size, config):
    super(SinoidGenerator, self).__init__(size, config)

  def texture_mask(self):
    first_phase = self.config['first_phase']
    frequency = self.config['frequency']
    height_offset = self.config['height_offset']
    amplitude = self.config['amplitude']
    angle = self.config['angle']
    first_phase = parse(first_phase)
    frequency =  parse(frequency)
    height_offset =  parse(height_offset)
    amplitude = parse(amplitude)
    angle = parse(angle)

    r = rotation_matrix(angle)
    X, Y = np.ogrid[:self.size[0], :self.size[1]]
    mask = height_offset + amplitude * (np.sin(frequency * (r@(np.array([X, Y])).T)[0] + first_phase)
                                                        + np.sin(frequency * (r@(np.array([X, Y])).T)[1] + first_phase)),
    # mask = np.fromfunction(lambda x, y:  height_offset + amplitude * (np.sin(frequency * (r@(np.array([x, y])).T)[0] + first_phase)
    #                                                     + np.sin(frequency * (r@(np.array([x, y])).T)[1] + first_phase)),
    #                        self.size)
    return mask[0]




