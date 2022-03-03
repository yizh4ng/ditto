class TextureGenerator():
  def __init__(self, size, config):
    self.size = size
    self.config = config

  def texture_mask(self):
    raise NotImplemented