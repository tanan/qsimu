import numpy as np
from scipy.optimize import LinearConstraint

def create_time_constraints(depth, random_list_num):
  matrix = np.array([])
  for i in range(depth):
    for j in range(depth):
      if j == (depth-1)-i:
        matrix = np.append(matrix, 1)
      elif j == (depth-1)-i-1:
        matrix = np.append(matrix, -1)
      else:
        matrix = np.append(matrix, 0)

  matrix = np.hstack((matrix.reshape(depth, depth), np.zeros((depth, random_list_num-depth))))
  print("time constraints matrix:", matrix)
  return LinearConstraint(matrix, 0, np.inf)