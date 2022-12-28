# coding: utf-8
import datetime
import sys
import time

import numpy as np
import yaml
from constraints import create_time_constraints
from hamiltonian import create_ising_hamiltonian
from qulacsvis import circuit_drawer
from random_list import randomize
from scipy.optimize import minimize

import qulacs
from qulacs import QuantumCircuit, QuantumState

sys.path.append("..")
import common.database.sqlite.job as sqlitejob
from common.ansatz.direct import DirectAnsatz
from common.ansatz.ising import IsingAnsatz
from common.ansatz.xy import XYAnsatz
from common.ansatz.xyz import XYZAnsatz
from common.database.bigquery import BigQueryClient
from common.database.bigquery.job_result import insert_job_result
from common.database.schema.job import JobFactory
from common.database.sqlite import DBClient
from common.optimizer import OptimizerStatus
from common.optimizer.adam import Adam
from common.measurement import sampling_indirect_measurement

## init variables
qulacs_hamiltonian = None
config = None
param_history = []
cost_history = []
iter_history = []
iteration = 0
ansatz = None
optimizer = Adam()


def init_ansatz(config):
    if config["gate"]["type"] == "indirect_xy":
        ansatz = XYAnsatz(
            config["nqubit"],
            config["depth"],
            config["gate"]["noise"],
            config["gate"]["parametric_rotation_gate_set"],
            config["gate"]["time"],
            config["gate"]["bn"],
        )
    elif config["gate"]["type"] == "indirect_xyz":
        ansatz = XYZAnsatz(
            config["nqubit"],
            config["depth"],
            config["gate"]["noise"],
            config["gate"]["parametric_rotation_gate_set"],
            config["gate"]["time"],
        )
    elif config["gate"]["type"] == "indirect_ising":
        ansatz = IsingAnsatz(
            config["nqubit"],
            config["depth"],
            config["gate"]["noise"],
            config["gate"]["parametric_rotation_gate_set"],
            config["gate"]["time"],
            config["gate"]["bn"],
        )
    else:
        ansatz = DirectAnsatz(
            config["nqubit"], config["depth"], config["gate"]["noise"]
        )
    return ansatz


def cost(params):
    global config
    global ansatz
    global qulacs_hamiltonian
    global iteration
    iteration += 1
    return estimator(config["nqubit"], ansatz, qulacs_hamiltonian, params)


def estimator(nqubit, ansatz, hamiltonian, params):
    state = QuantumState(nqubit)
    circuit = QuantumCircuit(nqubit)
    circuit = ansatz.create_ansatz(params)
    circuit.update_quantum_state(state)

    # return hamiltonian.get_expectation_value(state)
    return sampling_indirect_measurement(state, 1000)


def record(x):
    global param_history
    global cost_history
    global iter_history
    param_history.append(x)
    cost_history.append(cost(x))
    iter_history.append(iteration)
    print(cost_history)


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
    ## init qulacs hamiltonian
    global qulacs_hamiltonian
    qulacs_hamiltonian = create_ising_hamiltonian(config["nqubit"])

    ## init ansatz instance
    global ansatz
    ansatz = init_ansatz(config)

    ## randomize and create constraints
    init_random_list = [-7.26675583e-01, 9.47382121e-01, 1.87505152e+00,-3.19443372e+00, 1.46558207e+00,-2.26445091e+00, 1.32820896e+00,-2.29513217e+00, 2.80339263e+00, 4.19293214e-01,-1.17314854e+00,-1.47085879e+00, 6.75553695e-01, 4.70295281e-01, 1.93328703e-01,-4.37038780e-02, 5.30939632e-01, 7.38153266e-02, 5.84961669e-01,-2.03464719e-02,-1.59432718e-01, 2.21892083e-01,-3.41649594e-01,-2.35632631e-02, 1.38478139e+00,-1.66668537e-04, 3.85170573e-02, 4.51479897e-01, 1.66069783e+00,-4.13686628e-01, 1.25041576e+00,-6.34686100e-01,-1.36854193e+00,-2.16654503e-01, 5.41638398e-02,-3.46465510e-02, 5.20318430e-01, 2.87355773e-01, 6.75084537e-02, 3.79582671e-01, 6.76770820e-01,-3.73084514e-01,-5.86205096e-01, 4.94054163e-01, 6.77534564e-01, 7.54771801e-01, 1.10468163e+00, 6.67358824e-01,-2.47002175e-01,-5.04099017e-01,-6.16829671e-01, 6.87286585e-01,-3.12928671e-01,-3.31012959e-02,-6.44546729e-01,-6.97712873e-01,-9.53484048e-01, 7.53911637e-01,-1.00675977e+00, 3.77753020e-01]
    record(init_random_list)

    ## calculation
    # options = {"maxiter": 1000}
    # opt = minimize(
    #     cost, init_random_list, method=config["optimize"]["method"], options=options, callback=record
    # )
    # print(opt)


def start(config):
    reset()
    run(config)

if __name__ == "__main__":
    args = sys.argv
    path = args[1]
    iter_for_static = 1
    iter_for_random = 100
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        if (
            config["gate"]["type"] == ["direct", "indirect_xyz"]
            or config["gate"]["bn"]["type"] == "static"
        ):
            for k in range(iter_for_static):
                start(config)
