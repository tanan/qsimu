import numpy as np

def output(config, n_qubit, param_history, cost_history, iter_history):
  if config['gate']['type'] == 'direct':
    param_path = "results/%squbit_%s_%s_param.txt" % (n_qubit, config['gate']['type'], config['depth'])
    cost_path = "results/%squbit_%s_%s_cost.txt" % (n_qubit, config['gate']['type'], config['depth'])
    iter_path = "results/%squbit_%s_%s_iter.txt" % (n_qubit, config['gate']['type'], config['depth'])
  else:
    param_path = "results/%squbit_%s_%s_%s_%s_%s_%s_param.txt" % (n_qubit, config['gate']['type'], config['depth'], config['max_time'], config['gate']['is_r_random'], config['gate']['is_cn_random'], config['gate']['is_bn_random'])
    cost_path = "results/%squbit_%s_%s_%s_%s_%s_%s_cost.txt" % (n_qubit, config['gate']['type'], config['depth'], config['max_time'], config['gate']['is_r_random'], config['gate']['is_cn_random'], config['gate']['is_bn_random'])
    iter_path = "results/%squbit_%s_%s_%s_%s_%s_%s_iter.txt" % (n_qubit, config['gate']['type'], config['depth'], config['max_time'], config['gate']['is_r_random'], config['gate']['is_cn_random'], config['gate']['is_bn_random'])

  print(param_history)
  print(cost_history)
  print(iter_history)
  np.savetxt(param_path, param_history)
  np.savetxt(cost_path, cost_history)
  np.savetxt(iter_path, iter_history)