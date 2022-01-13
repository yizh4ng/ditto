import numpy as np
class Shape():
  def __init__(self):
    self.mask = None

  def move(self, offset):
    W, H = self.mask.shape[0], self.mask.shape[1]
    self.mask = np.pad(self.mask, ((W, W), (H, H)), 'constant', constant_values=(0, 0))
    self.mask = np.roll(self.mask, shift=offset,axis=(0, 1))
    self.mask = self.mask[W:-W, H:-H]

  def rotate(self, angle):
    h = np.max(self.mask)
    index = np.array(np.where(self.mask != 0))
    index = index.astype(int)

    c_w = np.mean(index[0, :]).astype(int)
    c_h = np.mean(index[1, :]).astype(int)
    index -= np.array([[c_w], [c_h]])
    theta = angle/(2 * np.pi)
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    index = np.dot(rotate_matrix, index)
    index += np.array([[c_w], [c_h]])
    index = index.astype(int)
    col_max, col_min = np.amax(index[0]), np.amin(index[0])
    new_mask = np.zeros(self.mask.shape).astype(int)
    for i in range(col_min, col_max-1):
      col_index = index[1, (index[0] == i)]
      if len(col_index) == 0: continue
      row_max, row_min = np.amax(col_index), np.amin(col_index)
      new_mask[i, np.clip(row_min, 0, self.mask.shape[0] - 1):np.clip(row_max,0, self.mask.shape[1])] = 1

    self.mask = new_mask * h
    pass

  def draw(self, img, draw_over=True):
    if draw_over:
      img = img * (self.mask==0) + self.mask
    else:
      img += self.mask
    assert isinstance(img, np.ndarray)
    return img
