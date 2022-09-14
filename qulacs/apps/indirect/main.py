# coding: utf-8
from cmath import cos
import sys
import yaml
from datetime import datetime as dt

sys.path.append('..')
from indirect.ansatz import *
from indirect.init_random import *
from indirect.init_hamiltonian import *
from indirect.constraints import *
from mimetypes import init
from qulacs import QuantumState, QuantumCircuit
from scipy.optimize import minimize, Bounds

def cost(random_list):
  global iteration
  iteration+=1
  state = QuantumState(n_qubit)
  circuit = QuantumCircuit(n_qubit)
  if config['gate']['type'] == 'indirect_by_none':
    ansatz = AnsatzIndirectByNone(n_qubit, random_list, config['depth'])
  elif config['gate']['type'] == 'indirect_by_xy_hamiltonian':
    ansatz = AnsatzIndirectByXY(n_qubit, random_list, config['depth'], config['gate']['is_bn_random'])
  elif config['gate']['type'] == 'indirect_by_xyz_hamiltonian':
    ansatz = AnsatzIndirectByXYZ(n_qubit, random_list, config['depth'])
  elif config['gate']['type'] == 'indirect_by_ising_hamiltonian':
    ansatz = AnsatzIndirectByIsing(n_qubit, random_list, config['depth'], config['gate']['is_bn_random'])
  else:
    ansatz = AnsatzDirect(n_qubit, random_list, config['depth'])

  circuit = ansatz.create_ansatz()  
  circuit.update_quantum_state(state)
  return qulacs_hamiltonian.get_expectation_value(state)

def record(x):
  global param_history
  global cost_history
  global iter_history
  param_history.append(x)
  cost_history.append(cost(x))
  iter_history.append(iteration)

def output(param_history, cost_history, iter_history):
  tdatetime = dt.now()
  tstr = tdatetime.strftime('%Y%m%d%H%M')
  if config['gate']['type'] == 'direct':
    param_path = "results/%squbit_%s_%s_param.txt" % (n_qubit, config['gate']['type'], config['depth'])
    cost_path = "results/%squbit_%s_%s_cost.txt" % (n_qubit, config['gate']['type'], config['depth'])
    iter_path = "results/%squbit_%s_%s_iter.txt" % (n_qubit, config['gate']['type'], config['depth'])
  else:
    param_path = "results/%squbit_%s_%s_%s_%s_%s_%s_param_%s.txt" % (n_qubit, config['gate']['type'], config['depth'], config['max_time'], config['gate']['is_r_random'], config['gate']['is_cn_random'], config['gate']['is_bn_random'], tstr)
    cost_path = "results/%squbit_%s_%s_%s_%s_%s_%s_cost_%s.txt" % (n_qubit, config['gate']['type'], config['depth'], config['max_time'], config['gate']['is_r_random'], config['gate']['is_cn_random'], config['gate']['is_bn_random'], tstr)
    iter_path = "results/%squbit_%s_%s_%s_%s_%s_%s_iter_%s.txt" % (n_qubit, config['gate']['type'], config['depth'], config['max_time'], config['gate']['is_r_random'], config['gate']['is_cn_random'], config['gate']['is_bn_random'], tstr)

  print(param_history)
  print(cost_history)
  print(iter_history)
  np.savetxt(param_path, param_history)
  np.savetxt(cost_path, cost_history)
  np.savetxt(iter_path, iter_history)

def run():
  init_random_list, bounds = randomize(n_qubit, config)
  if config['gate']['type'] != 'direct':
    constraints = create_time_constraints(config['depth'], len(init_random_list))
  record(init_random_list)
  # options = {"disp": True, "maxiter": 50, "gtol": 1e-600}
  if config['gate']['type'] == 'direct':
    opt = minimize(cost, init_random_list,
                method="BFGS",
                callback=record)
  else:
    opt = minimize(cost, init_random_list,
                  method="SLSQP",
                  constraints=constraints,
                  bounds=bounds,
                  callback=record)
  output(param_history, cost_history, iter_history)

## init
n_qubit = 6
qulacs_hamiltonian = init_hamiltonian()
config = {}
param_history = []
cost_history = []
iter_history = []
iteration = 0

if __name__ == '__main__':
  args = sys.argv
  path = args[1]
  with open(path, 'r') as f:
    config = yaml.safe_load(f)
    run()