# coding: utf-8
from cmath import cos
import sys
import yaml

sys.path.append('..')
from vqe_ising.ansatz import *
from vqe_ising.random_list import randomize
from vqe_ising.hamiltonian import create_ising_hamiltonian
from vqe_ising.constraints import create_time_constraints
from vqe_ising.output import output
from qulacs import QuantumState, QuantumCircuit
from scipy.optimize import minimize

def cost(random_list):
  global iteration
  iteration+=1
  state = QuantumState(n_qubit)
  circuit = QuantumCircuit(n_qubit)
  if config['gate']['type'] == 'indirect_by_none':
    ansatz = AnsatzIndirectByNone(n_qubit, random_list, config['depth'], config['gate']['parametric_rotation_gate_set'])
  elif config['gate']['type'] == 'indirect_by_xy_hamiltonian':
    ansatz = AnsatzIndirectByXY(n_qubit, random_list, config['depth'], config['gate']['parametric_rotation_gate_set'], config['gate']['is_bn_random'])
  elif config['gate']['type'] == 'indirect_by_xyz_hamiltonian':
    ansatz = AnsatzIndirectByXYZ(n_qubit, random_list, config['depth'], config['gate']['parametric_rotation_gate_set'])
  elif config['gate']['type'] == 'indirect_by_ising_hamiltonian':
    ansatz = AnsatzIndirectByIsing(n_qubit, random_list, config['depth'], config['gate']['parametric_rotation_gate_set'], config['gate']['is_bn_random'])
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
  output(config, n_qubit, param_history, cost_history, iter_history)

## init
n_qubit = 6
qulacs_hamiltonian = create_ising_hamiltonian(n_qubit)
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