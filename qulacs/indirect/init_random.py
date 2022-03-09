import numpy as np
import random

def randomize(n_qubit, config):
  '''
  Create random list for a parametric ciruit.
  Parameters are time, cn, r(gamma), bn
  time: 0 - max_time
  cn: 0 - 1
  r(gamma): 0 - 1
  bn: 0 - 1
  theta: 0 - 1
  Return [t1, t2, ... td, cn1, cn2, ... cnd, r1, r2, ..., rd, bn1, bn2, ..., bnd, theta1, ... theatad*gate_set]
  '''
  t = np.array([])
  # cn = np.array([])
  # r = np.array([])
  bn = np.array([])
  if config['gate']['type'] == "direct":
    return np.random.random(2*n_qubit*(config['depth']+1))*1e-1

  for i in range(config['depth']):
    t = np.append(t, random.uniform(0.0,config['max_time']))
    bn = np.append(bn, random.uniform(0.0, 1.0))
  init_random_list = []
  if config['gate']['is_bn_random']:
      init_random_list = np.append(t, bn)
  else:
    init_random_list = np.append(t, np.array([1]*config['depth']))
  
  init_random_list = np.append(init_random_list, np.random.random(config['gate']['parametric_rotation_gate_set']*config['depth'])*1e-1)
  return init_random_list