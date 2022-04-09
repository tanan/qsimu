import numpy as np
from scipy.optimize import Bounds

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
  # cn = np.array([])
  # r = np.array([])
  if config['gate']['type'] == "direct":
    list_count = 2*n_qubit*(config['depth']+1)
    return np.random.random(list_count)*1e-1, Bounds([-np.Inf]*list_count, [np.Inf]*list_count)

  init_random_list = np.random.uniform(0.0, config['max_time'], config['depth'])

  if config['gate']['is_bn_random'] and config['gate']['type'] != 'indirect_by_xyz_hamiltonian':
    init_random_list = np.append(init_random_list, np.random.uniform(0.0, 1.0, config['depth']*n_qubit))

  init_random_list = np.append(init_random_list, np.random.random(config['gate']['parametric_rotation_gate_set']*config['depth'])*1e-1)
  return init_random_list, get_bounds(n_qubit, config)

def get_bounds(n_qubit, config):
  t_min = np.array([0.0] * config['depth'])
  t_max = np.array([config['max_time']] * config['depth'])
  bn_min = np.array([0.0] * config['depth'] * n_qubit)
  bn_max = np.array([1.0] * config['depth'] * n_qubit)
  theta_min = np.array([-np.Inf] * (config['gate']['parametric_rotation_gate_set']*config['depth']))
  theta_max = np.array([np.Inf] * (config['gate']['parametric_rotation_gate_set']*config['depth']))

  if config['gate']['is_bn_random'] and config['gate']['type'] != 'indirect_by_xyz_hamiltonian':
    min_bounds = np.append(np.append(np.append(np.array([]), t_min), bn_min), theta_min)
    max_bounds = np.append(np.append(np.append(np.array([]), t_max), bn_max), theta_max)
  else:
    min_bounds = np.append(np.append(np.array([]), t_min), theta_min)
    max_bounds = np.append(np.append(np.array([]), t_max), theta_max)

  return Bounds(min_bounds, max_bounds)
