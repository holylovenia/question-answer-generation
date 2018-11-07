import numpy as np


class PageRank:

  def __init__(self, eps=0.0001, d=0.85):
    self.eps = eps
    self.d = d

  def page_rank(self, M):
    """
    similarity_matrix[i][j] = probability of transitioning from sentence i to sentence j
    eps = stop the algorithm when the difference between two consecutive iterations <= eps
    d (damping factor) = with a probability 1-d the user will simply pick a sentence at random as the next destination, ignoring the structure completely    
    """
    
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    last_v = np.ones((N, 1), dtype=np.float32) * 100
    M_hat = (self.d * M) + (((1 - self.d) / N) * np.ones((N, N), dtype=np.float32))

    while np.linalg.norm(v - last_v, 2) > self.eps:
      last_v = v
      v = np.matmul(M_hat, v)
    
    return v