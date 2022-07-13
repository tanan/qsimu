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
from random_list import randomize
from hamiltonian import create_ising_hamiltonian
from constraints import create_time_constraints
from output import to_string, output
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
  else:
    opt = minimize(cost, init_random_list,
                  method="SLSQP",
                  options=options,
                  callback=record)

  end_time = time.perf_counter()
  output(param_history, cost_history, iter_history)
  if config['gate']['type'] == 'direct':
    job = Job(
      now,
      end_time - start_time,
      config['nqubit'],
      config['depth'],
      config['gate']['type'],
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      str(cost_history[-1]),
      str(param_history[-1]),
      str(iter_history[-1]),
      str(cost_history),
      to_string(param_history),
      str(iter_history)
    )
  else:
    job = Job(
      now,
      end_time - start_time,
      config['nqubit'],
      config['depth'],
      config['gate']['type'],
      str(config['gate']['parametric_rotation_gate_set']),
      str(config['gate']['bn']['type']),
      config['gate']['bn']['range'] if 'range' in config['gate']['bn'] else None,
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
  client.insertJob(job)
  np.savetxt('data/xy_params.txt', param_history[-1])

if __name__ == '__main__':
  args = sys.argv
  path = args[1]
  with open(path, 'r') as f:
    config = yaml.safe_load(f)
    # non bn model
    if config['gate']['type'] == ['direct', 'indirect_xyz'] or config['gate']['bn']['type'] == 'static':
      for k in range(100):
        run()
        reset()
    else:
      for k in range(100):
        config['gate']['bn']['value'] = np.random.rand(config['nqubit']) * config['gate']['bn']['range'] - (config['gate']['bn']['range'] / 2)
        run()
        reset()
