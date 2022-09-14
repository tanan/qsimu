import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qulacs import QuantumState, QuantumCircuit, Observable

sys.path.append('..')
from common.ansatz.xy import XYAnsatz
from common.ansatz.direct import DirectAnsatz
from common.random_list import randomize

########  パラメータ  #############
time_step = 0.77  ## ランダムハミルトニアンによる時間発展の経過時間

## init variables
config = None
param_history = []
cost_history = []
# iter_history = []
# iteration = 0
ansatz = None

# target function
func_to_learn = lambda x: np.sin(x*np.pi)

## random seed
random_seed = 0
np.random.seed(random_seed)

def create_train_data(x_min, x_max, num_x_train=50, mag_noise=0.05):
  x_train = x_min + (x_max - x_min) * np.random.rand(num_x_train)
  y_train = func_to_learn(x_train) + mag_noise * np.random.randn(num_x_train)
  return x_train, y_train

# The gate for encoding x
def U_in(x, U_time):
  U = QuantumCircuit(config['nqubit'])
  angle_y = np.arcsin(x)
  angle_z = np.arccos(x**2)
  for i in range(0,1):
    U.add_RY_gate(i, angle_y)
    U.add_RZ_gate(i, angle_z)

  U.add_gate(U_time)

  return U

def qcl_pred(x, U_time, U_out):
  obs = Observable(config['nqubit'])
  obs.add_operator(2.,'Z 0')
  state = QuantumState(config['nqubit'])
  state.set_zero_state()

  U_in(x, U_time).update_quantum_state(state)
  U_out.update_quantum_state(state)
  res = obs.get_expectation_value(state)

  return res

def cost(random_list):
  global config
  global x_train
  global y_train
  global ansatz
  global time_step
  state = QuantumState(config['nqubit'])
  state.set_zero_state()

  U_out = ansatz.create_ansatz(random_list)

  y_pred = [qcl_pred(x, ansatz.create_hamiltonian_gate(time_step), U_out) for x in x_train]
  L = ((y_pred - y_train)**2).mean()
  return L

def create_ansatz(config):
  if config['gate']['bn']['type'] == "static_random":
    config['gate']['bn']['value'] = np.random.rand(config['nqubit']) * config['gate']['bn']['range'] - (config['gate']['bn']['range'] / 2)
  
  if config['gate']['type'] == 'indirect_xy':
    ansatz = XYAnsatz(config['nqubit'], config['depth'], config['gate']['parametric_rotation_gate_set'], config['gate']['time'], config['gate']['bn'])
  elif config['gate']['type'] == 'direct':
    ansatz = DirectAnsatz(config['nqubit'], config['depth'], config['gate']['parametric_rotation_gate_set'])
  return ansatz

def create_graph(U_time, U_out, x_min, x_max, y_init, x_train, y_train):
  plt.figure(figsize=(10, 6))
  xlist = np.arange(x_min, x_max, 0.02)
  plt.plot(x_train, y_train, "o", label='Teacher')
  plt.plot(xlist, y_init, '--', label='Initial Model Prediction', c='gray')
  y_pred = np.array([qcl_pred(x, U_time, U_out) for x in xlist])
  plt.plot(xlist, y_pred, label='Final Model Prediction')
  plt.legend()
  plt.show()

def record(x):
  global param_history
  global cost_history
  param_history.append(x)
  cost_history.append(cost(x))
  print(cost(x))

if __name__ == '__main__':
  args = sys.argv
  path = args[1]
  with open(path, 'r') as f:
    config = yaml.safe_load(f)

    ## create training data
    x_min = - 1.; x_max = 1.
    x_train, y_train = create_train_data(x_min, x_max)

    ## create Unitary gate instance
    ansatz = create_ansatz(config)

    ## init random list
    random_list, bounds = randomize(config['nqubit'], config)

    ## save init y
    U_time = ansatz.create_hamiltonian_gate(time_step)
    U_out = ansatz.create_ansatz(random_list)
    xlist = np.arange(x_min, x_max, 0.02)
    y_init = [qcl_pred(x, U_time, U_out) for x in xlist]

    # minimize
    result = minimize(cost, random_list, method='Nelder-Mead', callback=record)
    print(result)

    U_out = ansatz.create_ansatz(result.x)
    create_graph(U_time, U_out, x_min, x_max, y_init, x_train, y_train)


