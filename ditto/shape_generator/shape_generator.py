import numpy as np
from ditto.shape_generator import shape_dict


class BaseGenerator():
  _name = None

  def __init__(self):
    self.counter = 0
    self.num = 0
    self.height_range = None


  def get_random_value(self, intervals):
    if not isinstance(intervals[0], list):
      intervals = [intervals]

    height_candidates = []
    for interval in intervals:
      assert len(interval) == 2
      if interval[0] == interval[1]:
        height_candidates.append(interval[0])
      else:
        height_candidates.append(np.random.uniform(*interval))

    random_choise = np.random.randint(0, len(height_candidates))

    return height_candidates[random_choise]


  def generate(self, img):
    raise NotImplementedError
    pass

  def generate_one(self, img):
    raise NotImplementedError
    pass

  @classmethod
  def register(cls):
    shape_dict.update({cls._name: cls})