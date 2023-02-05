# coding: utf-8
import datetime
import sys
import time

import numpy as np
import yaml
from hamiltonian import create_ising_hamiltonian

import qulacs
from qulacs import QuantumCircuit, QuantumState

sys.path.append("..")
from common.ansatz.direct import DirectAnsatz
from common.ansatz.xy import XYAnsatz

## init variables
qulacs_hamiltonian = None
config = None
ansatz = None


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
    else:
        ansatz = DirectAnsatz(
            config["nqubit"], config["depth"], config["gate"]["noise"]
        )
    return ansatz


def cost(params):
    global config
    global ansatz
    global qulacs_hamiltonian
    return estimator(config["nqubit"], ansatz, qulacs_hamiltonian, params)


def estimator(nqubit, ansatz, hamiltonian, params):
    state = QuantumState(nqubit)
    circuit = QuantumCircuit(nqubit)
    circuit = ansatz.create_ansatz(params)
    circuit.update_quantum_state(state)

    return hamiltonian.get_expectation_value(state)


def run(config):
    ## init qulacs hamiltonian
    global qulacs_hamiltonian
    qulacs_hamiltonian = create_ising_hamiltonian(config["nqubit"])

    ## init ansatz instance
    global ansatz
    ansatz = init_ansatz(config)

    params = config['gate']['params']
    print(f"cost: ${cost(params)}")


if __name__ == "__main__":
    args = sys.argv
    path = args[1]
    iter_num = 10
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        for k in range(iter_num):
            run(config)
