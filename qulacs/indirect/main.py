# coding: utf-8
import sys
sys.path.append('..')
from indirect.ansatz import *
from mimetypes import init
import qulacs
from qulacs import QuantumState, QuantumCircuit
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.ops import QubitOperator
from scipy.optimize import minimize
import numpy as np
import random

def cost(random_list):
  state = QuantumState(n_qubit)
  circuit = QuantumCircuit(n_qubit)
  if gate == 'indirect_by_xy_hamiltonian':
    ansatz = AnsatzIndirectByXY(n_qubit, random_list, depth)
    circuit = ansatz.create_ansatz(cn, r, bn)
  elif gate == 'indirect_by_xyz_hamiltonian':
    ansatz = AnsatzIndirectByXYZ(n_qubit, random_list, depth)
    circuit = ansatz.create_ansatz(cn)
  else:
    ansatz = AnsatzDirect(n_qubit, random_list, depth)
    circuit = ansatz.create_ansatz()
  
  circuit.update_quantum_state(state)
  return qulacs_hamiltonian.get_expectation_value(state)

def init_hamiltonian():
  jw_hamiltonian = (-0.143021 * QubitOperator('Z0')) + (0.104962 * QubitOperator('Z0 Z1')) + (0.038195 * QubitOperator('Z1 Z2')) + (-0.325651 * QubitOperator('Z2')) + (-0.143021 * QubitOperator('Z3')) + (0.104962 * QubitOperator('Z3 Z4')) + (0.038195 * QubitOperator('Z4 Z5')) + (-0.325651 * QubitOperator('Z5')) + (0.172191 * QubitOperator('Z1')) + (0.174763 * QubitOperator('Z0 Z1 Z2')) + (0.136055 * QubitOperator('Z0 Z2')) + (0.116134 * QubitOperator('Z0 Z3')) + (0.094064 * QubitOperator('Z0 Z3 Z4')) + (0.099152 * QubitOperator('Z0 Z4 Z5')) + (0.123367 * QubitOperator('Z0 Z5')) + (0.094064 * QubitOperator('Z0 Z1 Z3')) + (0.098003 * QubitOperator('Z0 Z1 Z3 Z4')) + (0.102525 * QubitOperator('Z0 Z1 Z4 Z5')) + (0.097795 * QubitOperator('Z0 Z1 Z5')) + (0.099152 * QubitOperator('Z1 Z2 Z3')) + (0.102525 * QubitOperator('Z1 Z2 Z3 Z4')) + (0.112045 * QubitOperator('Z1 Z2 Z4 Z5')) + (0.105708 * QubitOperator('Z1 Z2 Z5')) + (0.123367 * QubitOperator('Z2 Z3')) + (0.097795 * QubitOperator('Z2 Z3 Z4')) + (0.105708 * QubitOperator('Z2 Z4 Z5')) + (0.133557 * QubitOperator('Z2 Z5')) + (0.172191 * QubitOperator('Z4')) + (0.174763 * QubitOperator('Z3 Z4 Z5')) + (0.136055 * QubitOperator('Z3 Z5')) + (0.059110 * QubitOperator('X0 Z1')) + (-0.059110 * QubitOperator('X0')) + (0.161019 * QubitOperator('Z1 X2')) + (-0.161019 * QubitOperator('X2')) + (0.059110 * QubitOperator('X3 Z4')) + (-0.059110 * QubitOperator('X3')) + (0.161019 * QubitOperator('Z4 X5')) + (-0.161019 * QubitOperator('X5')) + (-0.038098 * QubitOperator('X0 X2')) + (-0.003300 * QubitOperator('X0 Z1 X2')) + (0.013745 * QubitOperator('X0 Z1 X3 Z4')) + (-0.013745 * QubitOperator('X0 Z1 X3')) + (-0.013745 * QubitOperator('X0 X3 Z4')) + (0.013745 * QubitOperator('X0 X3')) + (0.011986 * QubitOperator('X0 Z1 Z4 X5')) + (-0.011986 * QubitOperator('X0 Z1 X5')) + (-0.011986 * QubitOperator('X0 Z4 X5')) + (0.011986 * QubitOperator('X0 X5')) + (0.011986 * QubitOperator('Z1 X2 X3 Z4')) + (-0.011986 * QubitOperator('Z1 X2 X3')) + (-0.011986 * QubitOperator('X2 X3 Z4')) + (0.011986 * QubitOperator('X2 X3')) + (0.013836 * QubitOperator('Z1 X2 Z4 X5')) + (-0.013836 * QubitOperator('Z1 X2 X5')) + (-0.013836 * QubitOperator('X2 Z4 X5')) + (0.013836 * QubitOperator('X2 X5')) + (-0.038098 * QubitOperator('X3 X5')) + (-0.003300 * QubitOperator('X3 Z4 X5')) + (-0.002246 * QubitOperator('Z0 Z1 X2')) + (0.002246 * QubitOperator('Z0 X2')) + (0.014815 * QubitOperator('Z0 X3 Z4')) + (-0.014815 * QubitOperator('Z0 X3')) + (0.009922 * QubitOperator('Z0 Z4 X5')) + (-0.009922 * QubitOperator('Z0 X5')) + (-0.002038 * QubitOperator('Z0 Z1 X3 Z4')) + (0.002038 * QubitOperator('Z0 Z1 X3')) + (-0.007016 * QubitOperator('Z0 Z1 Z4 X5')) + (0.007016 * QubitOperator('Z0 Z1 X5')) + (-0.006154 * QubitOperator('X0 Z2')) + (0.006154 * QubitOperator('X0 Z1 Z2')) + (0.014815 * QubitOperator('X0 Z1 Z3')) + (-0.014815 * QubitOperator('X0 Z3')) + (-0.002038 * QubitOperator('X0 Z1 Z3 Z4')) + (0.002038 * QubitOperator('X0 Z3 Z4')) + (0.001124 * QubitOperator('X0 Z1 Z4 Z5')) + (-0.001124 * QubitOperator('X0 Z4 Z5')) + (0.017678 * QubitOperator('X0 Z1 Z5')) + (-0.017678 * QubitOperator('X0 Z5')) + (-0.041398 * QubitOperator('Y0 Y2')) + (0.011583 * QubitOperator('Y0 Y1 X3 X4 Z5')) + (-0.011094 * QubitOperator('Y0 Y1 X4')) + (0.010336 * QubitOperator('Y1 Y2 X3 X4 Z5')) + (-0.005725 * QubitOperator('Y1 Y2 X4')) + (-0.006154 * QubitOperator('X3 Z5')) + (0.011583 * QubitOperator('X0 X1 Z2 X3 X4 Z5')) + (-0.011094 * QubitOperator('X0 X1 Z2 X4')) + (-0.011094 * QubitOperator('X1 X3 X4 Z5')) + (0.026631 * QubitOperator('X1 X4')) + (-0.017678 * QubitOperator('Z2 X3')) + (0.011583 * QubitOperator('X0 X1 Z2 Y3 Y4')) + (0.010336 * QubitOperator('X0 X1 Z2 Y4 Y5')) + (-0.011094 * QubitOperator('X1 Y3 Y4')) + (-0.005725 * QubitOperator('X1 Y4 Y5')) + (-0.041398 * QubitOperator('Y3 Y5')) + (0.011583 * QubitOperator('Y0 Y1 Y3 Y4')) + (0.010336 * QubitOperator('Y0 Y1 Y4 Y5')) + (0.010336 * QubitOperator('Y1 Y2 Y3 Y4')) + (0.010600 * QubitOperator('Y1 Y2 Y4 Y5')) + (0.024909 * QubitOperator('X0 X1 Z2 X3 X4 X5')) + (-0.031035 * QubitOperator('X1 X3 X4 X5')) + (-0.010064 * QubitOperator('Z2 X5')) + (0.024909 * QubitOperator('X0 X1 Z2 Y3 X4 Y5')) + (-0.031035 * QubitOperator('X1 Y3 X4 Y5')) + (0.024909 * QubitOperator('Y0 Y1 X3 X4 X5')) + (0.021494 * QubitOperator('Y1 Y2 X3 X4 X5')) + (0.024909 * QubitOperator('Y0 Y1 Y3 X4 Y5')) + (0.021494 * QubitOperator('Y1 Y2 Y3 X4 Y5')) + (0.011094 * QubitOperator('X0 X1 Z2 Z3 X4 Z5')) + (-0.026631 * QubitOperator('X1 Z3 X4 Z5')) + (0.011094 * QubitOperator('Y0 Y1 Z3 X4 Z5')) + (0.005725 * QubitOperator('Y1 Y2 Z3 X4 Z5')) + (0.010336 * QubitOperator('X0 X1 Z2 Z3 X4 X5')) + (-0.005725 * QubitOperator('X1 Z3 X4 X5')) + (0.002246 * QubitOperator('Z3 X5')) + (0.010336 * QubitOperator('Y0 Y1 Z3 X4 X5')) + (0.010600 * QubitOperator('Y1 Y2 Z3 X4 X5')) + (0.024909 * QubitOperator('X0 X1 X2 X3 X4 Z5')) + (-0.031035 * QubitOperator('X0 X1 X2 X4')) + (-0.010064 * QubitOperator('X2 Z5')) + (0.024909 * QubitOperator('X0 X1 X2 Y3 Y4')) + (0.021494 * QubitOperator('X0 X1 X2 Y4 Y5')) + (0.024909 * QubitOperator('Y0 X1 Y2 X3 X4 Z5')) + (-0.031035 * QubitOperator('Y0 X1 Y2 X4')) + (0.024909 * QubitOperator('Y0 X1 Y2 Y3 Y4')) + (0.021494 * QubitOperator('Y0 X1 Y2 Y4 Y5')) + (0.063207 * QubitOperator('X0 X1 X2 X3 X4 X5')) + (0.063207 * QubitOperator('X0 X1 X2 Y3 X4 Y5')) + (0.063207 * QubitOperator('Y0 X1 Y2 X3 X4 X5')) + (0.063207 * QubitOperator('Y0 X1 Y2 Y3 X4 Y5')) + (0.031035 * QubitOperator('X0 X1 X2 Z3 X4 Z5')) + (-0.009922 * QubitOperator('X2 Z3')) + (0.031035 * QubitOperator('Y0 X1 Y2 Z3 X4 Z5')) + (0.021494 * QubitOperator('X0 X1 X2 Z3 X4 X5')) + (0.021494 * QubitOperator('Y0 X1 Y2 Z3 X4 X5')) + (0.011094 * QubitOperator('Z0 X1 Z2 X3 X4 Z5')) + (-0.026631 * QubitOperator('Z0 X1 Z2 X4')) + (0.011094 * QubitOperator('Z0 X1 Z2 Y3 Y4')) + (0.005725 * QubitOperator('Z0 X1 Z2 Y4 Y5')) + (0.031035 * QubitOperator('Z0 X1 Z2 X3 X4 X5')) + (0.031035 * QubitOperator('Z0 X1 Z2 Y3 X4 Y5')) + (0.026631 * QubitOperator('Z0 X1 Z2 Z3 X4 Z5')) + (0.005725 * QubitOperator('Z0 X1 Z2 Z3 X4 X5')) + (0.010336 * QubitOperator('Z0 X1 X2 X3 X4 Z5')) + (-0.005725 * QubitOperator('Z0 X1 X2 X4')) + (0.010336 * QubitOperator('Z0 X1 X2 Y3 Y4')) + (0.010600 * QubitOperator('Z0 X1 X2 Y4 Y5')) + (0.021494 * QubitOperator('Z0 X1 X2 X3 X4 X5')) + (0.021494 * QubitOperator('Z0 X1 X2 Y3 X4 Y5')) + (0.005725 * QubitOperator('Z0 X1 X2 Z3 X4 Z5')) + (0.010600 * QubitOperator('Z0 X1 X2 Z3 X4 X5')) + (0.001124 * QubitOperator('Z1 Z2 X3 Z4')) + (-0.001124 * QubitOperator('Z1 Z2 X3')) + (-0.007952 * QubitOperator('Z1 Z2 Z4 X5')) + (0.007952 * QubitOperator('Z1 Z2 X5')) + (0.017678 * QubitOperator('Z2 X3 Z4')) + (0.010064 * QubitOperator('Z2 Z4 X5')) + (0.009922 * QubitOperator('Z1 X2 Z3')) + (-0.007016 * QubitOperator('Z1 X2 Z3 Z4')) + (0.007016 * QubitOperator('X2 Z3 Z4')) + (-0.007952 * QubitOperator('Z1 X2 Z4 Z5')) + (0.007952 * QubitOperator('X2 Z4 Z5')) + (0.010064 * QubitOperator('Z1 X2 Z5')) + (-0.002246 * QubitOperator('Z3 Z4 X5')) + (0.006154 * QubitOperator('X3 Z4 Z5'))
  return create_observable_from_openfermion_text(str(jw_hamiltonian))

def run(t_range):
  cost_history = []
  t = np.array([])
  for i in range(depth):
    t = np.append(t, random.uniform(0.0,t_range))
  init_random_list = np.append(t, np.random.random(4*depth)*1e-1)
  if gate == "direct":
      init_random_list = np.random.random(2*n_qubit*(depth+1))*1e-1

  cost_history.append(cost(init_random_list))
  method = "BFGS"
  options = {"disp": True, "maxiter": 50, "gtol": 1e-600}
  opt = minimize(cost, init_random_list,
                method=method,
                callback=lambda x: cost_history.append(cost(x)))
  filename = '%_%_%.txt' % (gate, depth, t_range)
  f = open(filename, 'w')
  f.write(cost_history)

## init
n_qubit = 6
depth = n_qubit * 2
cn = [1] * n_qubit
r = 0
bn = [0] * n_qubit
gate = "direct"
qulacs_hamiltonian = init_hamiltonian()

if __name__ == '__main__':
  args = sys.argv
  gate = args[1]
  depth = int(args[2])
  t_range = float(args[3])
  run(t_range)