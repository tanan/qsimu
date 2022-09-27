# coding: utf-8
import sys
import yaml
import time
import datetime
import numpy as np

from ansatz.xy import XYAnsatz
from ansatz.xyz import XYZAnsatz
from ansatz.ising import IsingAnsatz
from ansatz.direct import DirectAnsatz
from job_factory import JobFactory
from random_list import randomize
from hamiltonian import create_ising_hamiltonian
from constraints import create_time_constraints
from utils import to_string, output
from job import Job
from dbclient import DBClient
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

def init_ansatz(config):
  if config['gate']['type'] == 'indirect_xy':
    ansatz = XYAnsatz(config['nqubit'], config['depth'], config['gate']['noise'], config['gate']['parametric_rotation_gate_set'], config['gate']['time'], config['gate']['bn'])
  elif config['gate']['type'] == 'indirect_xyz':
    ansatz = XYZAnsatz(config['nqubit'], config['depth'], config['gate']['noise'], config['gate']['parametric_rotation_gate_set'], config['gate']['time'])
  elif config['gate']['type'] == 'indirect_ising':
    ansatz = IsingAnsatz(config['nqubit'], config['depth'], config['gate']['noise'], config['gate']['parametric_rotation_gate_set'], config['gate']['time'], config['gate']['bn'])
  else:
    ansatz = DirectAnsatz(config['nqubit'], config['depth'], config['gate']['noise'])
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

def run(config):
  ## performance measurement
  start_time = time.perf_counter()
  now = datetime.datetime.now()

  ## init qulacs hamiltonian
  global qulacs_hamiltonian
  qulacs_hamiltonian = create_ising_hamiltonian(config['nqubit'])

  ## init ansatz instance
  global ansatz
  ansatz = init_ansatz(config)

  ## randomize and create constraints
  init_random_list, bounds = randomize(config['nqubit'], config)
  record(init_random_list)

  ## calculation
  options = { 'maxiter' : 1000 }
  if config['gate']['constraints']:
    constraints = create_time_constraints(config['depth'], len(init_random_list))
    opt = minimize(cost, init_random_list,
                  method="SLSQP",
                  constraints=constraints,
                  bounds=bounds,
                  options=options,
                  callback=record)
  elif config['gate']['bounds']:
    opt = minimize(cost, init_random_list,
                  method="SLSQP",
                  options=options,
                  bounds=bounds,
                  callback=record)
  else:
    opt = minimize(cost, init_random_list,
                  method="SLSQP",
                  options=options,
                  callback=record)

  end_time = time.perf_counter()

  ## record to database
  job = JobFactory(config).create(now, start_time, end_time, cost_history, param_history, iter_history)
  client = DBClient("data/job_results.sqlite3")
  client.insertJob(job)
  # output(param_history, cost_history, iter_history)
  # np.savetxt('data/xy_params.txt', param_history[-1])

def start(config):
  reset()
  run(config)

if __name__ == '__main__':
  args = sys.argv
  path = args[1]
  iter_for_static = 10
  iter_for_random = 100
  with open(path, 'r') as f:
    config = yaml.safe_load(f)
    if config['gate']['type'] == ['direct', 'indirect_xyz'] or config['gate']['bn']['type'] == 'static':
      for k in range(iter_for_static):
        start(config)
    else:
      for k in range(iter_for_random):
        config['gate']['bn']['value'] = np.random.rand(config['nqubit']) * config['gate']['bn']['range'] - (config['gate']['bn']['range'] / 2)
        start(config)
