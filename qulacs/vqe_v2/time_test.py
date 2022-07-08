# coding: utf-8
from mimetypes import init
import sys
import yaml
import numpy as np

sys.path.append('..')
from vqe_v2.ansatz.xy import XYAnsatz
from vqe_v2.ansatz.xyz import XYZAnsatz
from vqe_v2.ansatz.ising import IsingAnsatz
from vqe_v2.ansatz.direct import DirectAnsatz
from vqe_v2.hamiltonian import create_ising_hamiltonian
from qulacs import QuantumState, QuantumCircuit

## init variables
qulacs_hamiltonian = None
config = None
ansatz = None

def init_ansatz():
  global config
  if config['gate']['type'] == 'indirect_xy':
    ansatz = XYAnsatz(config['nqubit'], config['depth'], config['gate']['parametric_rotation_gate_set'], config['gate']['bn'])
  elif config['gate']['type'] == 'indirect_xyz':
    ansatz = XYZAnsatz(config['nqubit'], config['depth'], config['gate']['parametric_rotation_gate_set'])
  elif config['gate']['type'] == 'indirect_ising':
    ansatz = IsingAnsatz(config['nqubit'], config['depth'], config['gate']['parametric_rotation_gate_set'], config['gate']['bn'])
  else:
    ansatz = DirectAnsatz(config['nqubit'], config['depth'])
  return ansatz

def cost(random_list):
  global config
  global ansatz
  state = QuantumState(config['nqubit'])
  circuit = QuantumCircuit(config['nqubit'])
  circuit = ansatz.create_ansatz(random_list)
  circuit.update_quantum_state(state)
  
  global qulacs_hamiltonian
  return qulacs_hamiltonian.get_expectation_value(state)

def run(init_random_list):
  global config
  ## init qulacs hamiltonian
  global qulacs_hamiltonian
  qulacs_hamiltonian = create_ising_hamiltonian(config['nqubit'])

  ## init ansatz instance
  global ansatz
  ansatz = init_ansatz()

  ## calculation
  print(init_random_list)
  print(cost(init_random_list))
  for k in [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:
    print(cost(np.append(k, init_random_list[1:])))

if __name__ == '__main__':
  args = sys.argv
  path = args[1]
  # params = args[2]
  # init_random_list = [ float(x) for x in params.replace(' ', '').strip('[').strip(']').strip().split(',') ]
  init_random_list = np.loadtxt('data/xy_params.txt')
  with open(path, 'r') as f:
    config = yaml.safe_load(f)
    run(init_random_list)
