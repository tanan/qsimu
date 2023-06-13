import sys
import yaml
from typing import Callable
import numpy as np
from scipy.optimize import minimize

from qulacs import Observable, QuantumCircuit, QuantumState

sys.path.append("..")
from common.utils.function import func_sin
from common.utils.graph import create_qcl_graph
from common.operator.operator import Operator
from common.ansatz import (
    Ansatz,
    DirectAnsatz,
    XYAnsatz,
)
from common.random_list import randomize


def create_train_data(x_min: float, x_max: float, num_x_train: int = 50, mag_noise: float = 0.05):
    x_train = x_min + (x_max - x_min) * np.random.rand(num_x_train)
    y_train = func_sin(x_train) + mag_noise * np.random.randn(num_x_train)
    return x_train, y_train


def init_ansatz(config: dict):
    if config["gate"]["bn"]["type"] == "static_random":
        config["gate"]["bn"]["value"] = np.random.rand(config["nqubit"]) * config[
            "gate"
        ]["bn"]["range"] - (config["gate"]["bn"]["range"] / 2)

    if config["gate"]["type"] == "indirect_xy":
        ansatz = XYAnsatz(
            config["nqubit"],
            config["depth"],
            config["gate"]["noise"],
            config["gate"]["parametric_rotation_gate_set"],
            config["gate"]["time"],
            config["gate"]["bn"],
        )
    elif config["gate"]["type"] == "direct":
        ansatz = DirectAnsatz(
            config["nqubit"],
            config["depth"],
            config["gate"]["noise"],
            config["gate"]["parametric_rotation_gate_set"],
        )
    return ansatz


def create_circuit_for_encoding(n_qubits: int):
    circuit = QuantumCircuit(n_qubits)

    def circuit_for_encoding(x: float):
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x**2)

        for i in range(2):
            circuit.add_RY_gate(i, angle_y)
            circuit.add_RZ_gate(i, angle_z)
    
        return circuit

    return circuit_for_encoding


def create_qcl_pred(n_qubits: int, op: Operator, u_in: Callable[[float], QuantumCircuit]):
    obs = Observable(n_qubits)
    obs.add_operator(op.coef, op.label)
    state = QuantumState(n_qubits)
    state.set_zero_state()

    def predict(x: float, u_out: QuantumCircuit):
        u_in(x).update_quantum_state(state)
        u_out.update_quantum_state(state)
        res = obs.get_expectation_value(state)

        return res

    return predict


def cost(params: any, qcl_pred: Callable[[float, Ansatz], float], ansatz: Ansatz, x_train: any, y_train: any):
    u_out = ansatz.create_ansatz(params)
    y_pred = [qcl_pred(x, u_out) for x in x_train]
    L = ((y_pred - y_train) ** 2).mean()
    return L


if __name__ == "__main__":
    args = sys.argv
    path = args[1]
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        n_qubits = config["nqubit"]

        ## create training data
        x_min = -1.0
        x_max = 1.0
        x_train, y_train = create_train_data(x_min, x_max)

        ## create ansatz instance
        ansatz = init_ansatz(config)

        ## init params
        init_params, bounds = randomize(n_qubits, config)

        ## prepare predict function
        op = Operator(2.0, "Z 0")
        u_in = create_circuit_for_encoding(n_qubits)
        qcl_pred = create_qcl_pred(n_qubits, op, u_in)

        ## save init y
        x_init = np.arange(x_min, x_max, 0.02)
        u_out = ansatz.create_ansatz(init_params)
        y_init = [qcl_pred(x, u_out) for x in x_init]

        ## prepare cost_fn
        def cost_fn(params):
            return cost(params, qcl_pred, ansatz, x_train, y_train)

        # minimize
        result = minimize(cost_fn, init_params, method="Nelder-Mead")

        # show results
        u_out = ansatz.create_ansatz(result.x)
        y_pred = [qcl_pred(x, u_out) for x in x_init]
        create_qcl_graph(x_init, y_init, x_train, y_train, y_pred)
