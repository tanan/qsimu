# coding: utf-8
import sys
import yaml
import time
import datetime
import numpy as np

sys.path.append("..")
from vqe_v2.ansatz.xy import XYAnsatz
from vqe_v2.ansatz.xyz import XYZAnsatz
from vqe_v2.ansatz.ising import IsingAnsatz
from vqe_v2.ansatz.direct import DirectAnsatz
from vqe_v2.random_list import randomize
from vqe_v2.hamiltonian import create_ising_hamiltonian
from vqe_v2.constraints import create_time_constraints
from vqe_v2.output import output
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
    if config["gate"]["type"] == "indirect_xy":
        ansatz = XYAnsatz(
            config["nqubit"],
            config["depth"],
            config["gate"]["parametric_rotation_gate_set"],
            config["gate"]["time"],
            config["gate"]["bn"],
        )
    elif config["gate"]["type"] == "indirect_xyz":
        ansatz = XYZAnsatz(
            config["nqubit"],
            config["depth"],
            config["gate"]["parametric_rotation_gate_set"],
        )
    elif config["gate"]["type"] == "indirect_ising":
        ansatz = IsingAnsatz(
            config["nqubit"],
            config["depth"],
            config["gate"]["parametric_rotation_gate_set"],
            config["gate"]["bn"],
        )
    else:
        ansatz = DirectAnsatz(config["nqubit"], config["depth"])
    return ansatz


def cost(random_list):
    global config
    global iteration
    global ansatz
    iteration += 1
    state = QuantumState(config["nqubit"])
    circuit = QuantumCircuit(config["nqubit"])
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
    qulacs_hamiltonian = create_ising_hamiltonian(config["nqubit"])

    ## init ansatz instance
    global ansatz
    ansatz = init_ansatz()

    ## randomize and create constraints
    init_random_list, bounds = randomize(config["nqubit"], config)

    ## temp
    init_random_list = np.loadtxt("data/xy_params_copy.txt")
    print(init_random_list)

    record(init_random_list)
    for k in np.arange(-20.0, 20.0, 0.1, dtype=float):
        # record(np.append(np.append(init_random_list[0], k), init_random_list[2:]))
        record(np.append(np.append(init_random_list[0:12], k), init_random_list[13:]))
        # record(np.append(k, init_random_list[1:]))

    print(cost_history)
    ## calculation
    # options = { 'maxiter' : 1000 }
    # if config['gate']['constraints']:
    #   constraints = create_time_constraints(config['depth'], len(init_random_list))
    #   opt = minimize(cost, init_random_list,
    #                 method="SLSQP",
    #                 constraints=constraints,
    #                 bounds=bounds,
    #                 options=options,
    #                 callback=record)
    # else:
    #   opt = minimize(cost, init_random_list,
    #                 method="SLSQP",
    #                 options=options,
    #                 callback=record)

    # end_time = time.perf_counter()
    # output(param_history, cost_history, iter_history)
    # np.savetxt('data/xy_params.txt', param_history[-1])


if __name__ == "__main__":
    args = sys.argv
    path = args[1]
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        # non bn model
        run()
        reset()
