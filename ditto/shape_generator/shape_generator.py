import numpy as np


class Shape_generator():

  def __init__(self):
    self.counter = 0
    self.num = 0
    self.height_range = None
    pass

  def get_random_height(self):
    height_candidates = []
    for interval in self.height_range:
      height_candidates.append(np.random.uniform(*interval))

    random_choise = np.random.randint(0, len(height_candidates))

    return height_candidates[random_choise]


  def generate(self, img):
    raise NotImplementedError
    pass

  def generate_one(self, img):
    raise NotImplementedError
    pass