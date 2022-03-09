# coding: utf-8
import sys
import yaml

sys.path.append('..')
from indirect.ansatz import *
from indirect.init_random import *
from indirect.init_hamiltonian import *
from mimetypes import init
from qulacs import QuantumState, QuantumCircuit
from scipy.optimize import minimize, Bounds

def cost(random_list):
  state = QuantumState(n_qubit)
  circuit = QuantumCircuit(n_qubit)
  if config['gate']['type'] == 'indirect_by_none':
    ansatz = AnsatzIndirectByNone(n_qubit, random_list, config['depth'])
  elif config['gate']['type'] == 'indirect_by_xy_hamiltonian':
    ansatz = AnsatzIndirectByXY(n_qubit, random_list, config['depth'])
  elif config['gate']['type'] == 'indirect_by_xyz_hamiltonian':
    ansatz = AnsatzIndirectByXYZ(n_qubit, random_list, config['depth'])
  elif config['gate']['type'] == 'indirect_by_ising_hamiltonian':
    ansatz = AnsatzIndirectByIsing(n_qubit, random_list, config['depth'])
  else:
    ansatz = AnsatzDirect(n_qubit, random_list, config['depth'])

  circuit = ansatz.create_ansatz()  
  circuit.update_quantum_state(state)
  return qulacs_hamiltonian.get_expectation_value(state)

def record(x):
  global param_history
  global cost_history
  param_history.append(x)
  cost_history.append(cost(x))

def run():
  init_random_list, bounds = randomize(n_qubit, config)
  global param_history
  global cost_history
  param_history.append(init_random_list)
  cost_history.append(cost(init_random_list))
  method = "SLSQP"
  # options = {"disp": True, "maxiter": 50, "gtol": 1e-600}
  opt = minimize(cost, init_random_list,
                method=method,
                bounds=bounds,
                callback=record)
  filepath = "results/%s_%s_%s.txt" % (config['gate']['type'], config['depth'], config['max_time'])
  f = open(filepath, 'w')
  print(param_history)
  print(cost_history)
  for v in cost_history:
    f.write(", %s" % str(v))

## init
n_qubit = 6
qulacs_hamiltonian = init_hamiltonian()
config = {}
param_history = []
cost_history = []

if __name__ == '__main__':
  args = sys.argv
  path = args[1]
  with open(path, 'r') as f:
    config = yaml.safe_load(f)
    run()