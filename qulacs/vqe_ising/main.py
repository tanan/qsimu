# coding: utf-8
from cmath import cos
from random import random
import sys
import yaml
import time
import datetime
import numpy as np

sys.path.append('..')
from vqe_ising.ansatz import AnsatzIndirectByXY, AnsatzIndirectByXYZ, AnsatzIndirectByIsing, AnsatzDirect
from vqe_ising.random_list import randomize
from vqe_ising.hamiltonian import create_ising_hamiltonian
from vqe_ising.constraints import create_time_constraints
from vqe_ising.output import to_string, output
from vqe_ising.job import Job
from vqe_ising.dbclient import DBClient
from qulacs import QuantumState, QuantumCircuit
from scipy.optimize import minimize

## init variables
qulacs_hamiltonian = None
config = None
param_history = []
cost_history = []
iter_history = []
iteration = 0
ansatz = None

def init_ansatz():
  global config
  if config['gate']['type'] == 'indirect_xy':
    ansatz = AnsatzIndirectByXY(config['nqubit'], config['depth'], config['gate']['parametric_rotation_gate_set'], config['gate']['bn'])
  elif config['gate']['type'] == 'indirect_xyz':
    ansatz = AnsatzIndirectByXYZ(config['nqubit'], config['depth'], config['gate']['parametric_rotation_gate_set'])
  elif config['gate']['type'] == 'indirect_ising':
    ansatz = AnsatzIndirectByIsing(config['nqubit'], config['depth'], config['gate']['parametric_rotation_gate_set'], config['gate']['bn'])
  else:
    ansatz = AnsatzDirect(config['nqubit'], config['depth'])
  return ansatz

def cost(random_list):
  global config
  global iteration
  global ansatz
  iteration+=1
  state = QuantumState(config['nqubit'])
  circuit = QuantumCircuit(config['nqubit'])
  circuit = ansatz.create_ansatz(random_list)  
  circuit.update_quantum_state(state)
  
  global qulacs_hamiltonian
  return qulacs_hamiltonian.get_expectation_value(state)

def record(x):
  global param_history
  global cost_history
  global iter_history
  param_history.append(x)
  cost_history.append(cost(x))
  iter_history.append(iteration)

def reset():
  global param_history
  global cost_history
  global iter_history
  global iteration
  param_history = []
  cost_history = []
  iter_history = []
  iteration = 0

def run():
  global config
  ## performance measurement
  start_time = time.perf_counter()
  now = datetime.datetime.now()

  ## init qulacs hamiltonian
  global qulacs_hamiltonian
  qulacs_hamiltonian = create_ising_hamiltonian(config['nqubit'])

  ## init ansatz instance
  global ansatz
  ansatz = init_ansatz()

  ## randomize and create constraints
  init_random_list, bounds = randomize(config['nqubit'], config)
  constraints = create_time_constraints(config['depth'], len(init_random_list))
  record(init_random_list)

  ## calculation
  options = { 'maxiter' : 300}
  opt = minimize(cost, init_random_list,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options=options,
                callback=record)

  output(param_history, cost_history, iter_history)
  end_time = time.perf_counter()
  job = Job(
    now,
    end_time - start_time,
    config['nqubit'],
    config['depth'],
    config['gate']['type'],
    str(config['gate']['parametric_rotation_gate_set']),
    str(config['gate']['bn']['type']),
    str(config['gate']['bn']['value']),
    str(config['gate']['cn']['value']),
    str(config['gate']['r']['value']),
    config['gate']['max_time'],
    str(cost_history[-1]),
    str(param_history[-1]),
    str(iter_history[-1]),
    str(cost_history),
    to_string(param_history),
    str(iter_history)
  )
  client = DBClient("data/job_results.sqlite3")
  client.insert(job)

if __name__ == '__main__':
  args = sys.argv
  path = args[1]
  with open(path, 'r') as f:
    config = yaml.safe_load(f)
    for k in range(10):
      for i in np.arange(-1.0, 1.0, 0.1):
        config['gate']['bn']['value'] = [i] * config['nqubit']
        run()
        reset()
