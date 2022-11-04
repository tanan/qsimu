import math
import sys
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize

from qulacs import Observable, QuantumCircuit, QuantumState

sys.path.append("..")
from common import hamiltonian
from common.ansatz.ansatz import AnsatzType
from common.ansatz.direct import DirectAnsatz
from common.ansatz.xy import XYAnsatz
from common.estimator import Observables
from common.estimator.hamiltonian import create_hamiltonian_estimator
from common.hamiltonian import HamiltonianModel
from common.hamiltonian.generator import \
    create_transverse_ising_hamiltonian_generator
from common.random_list import randomize

########  パラメータ  #############
time_step = 0.77  ## ランダムハミルトニアンによる時間発展の経過時間

## init variables
nqubit = None
config = None
# param_history = []
# cost_history = []
# iter_history = []
# iteration = 0
ansatz = None

# target function
func_to_learn = lambda x: np.sin(x * np.pi)

## random seed
random_seed = 0
np.random.seed(random_seed)


def create_ansatz(config):
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


def fully_connected_combinations_count(n):
    r = 2
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))


def U_in(nqubit, x, U_time):
    U = QuantumCircuit(nqubit)
    angle_y = np.arcsin(x)
    angle_z = np.arccos(x**2)
    for loop in range(3):
        for i in range(nqubit):
            U.add_RY_gate(i, angle_y)
            U.add_RZ_gate(i, angle_z)

        U.add_gate(U_time)
    return U


def create_train_data(
    nqubit: int, x_min: float = -1, x_max: float = 1, num_x_train: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    x_train = np.arange(x_min, x_max, num_x_train)
    observables = []
    for i in range(3):
        obs = Observable(nqubit)
        obs.add_operator(1.0, f"Z {i}")
        observables.append(obs)

    estimator = create_hamiltonian_estimator(
        nqubit,
        observables,
        (
            np.random.uniform(-1, 1, fully_connected_combinations_count(nqubit)),
            np.random.uniform(-1, 1, nqubit),
        ),
        HamiltonianModel.TRANSVERSE_ISING,
        x_train,
    )

    y_train = estimator.value

    return np.array(x_train), np.array(y_train)


def qcl_pred(nqubit, x, U_time, U_out):
    y = []
    for target in range(3):
        obs = Observable(nqubit)
        obs.add_operator(1.0, f"Z {target}")
        state = QuantumState(nqubit)
        state.set_zero_state()
        U_in(nqubit, x, U_time).update_quantum_state(state)
        U_out.update_quantum_state(state)
        y.append(obs.get_expectation_value(state))
    return y


def create_U_time(nqubit, ansatz, time_step):
    if ansatz.ansatz_type == AnsatzType.DIRECT:
        hamiltonian = create_transverse_ising_hamiltonian_generator(
            nqubit,
            np.random.uniform(-1, 1, fully_connected_combinations_count(nqubit)),
            np.random.uniform(-1, 1, nqubit),
            HamiltonianModel.TRANSVERSE_ISING,
        )
        return hamiltonian.time_evol_operator(time_step)
    else:
        return ansatz.create_hamiltonian_gate(time_step)


def cost(random_list):
    global nqubit
    global x_train
    global y_train
    global ansatz
    global time_step

    U_out = ansatz.create_ansatz(random_list)

    U_time = create_U_time(nqubit, ansatz, time_step)

    y_pred = np.array([qcl_pred(nqubit, x, U_time, U_out) for x in x_train])
    L = ((y_pred - y_train) ** 2).mean()
    return L


def create_graph(x_train, y_init, y_teacher, y_pred, name, nqubit, depth):
    plt.figure(figsize=(10, 6))
    plt.plot(x_train, y_teacher[0], ".", label="Teacher[0]")
    plt.plot(x_train, y_teacher[1], ".", label="Teacher[1]")
    plt.plot(x_train, y_teacher[2], ".", label="Teacher[2]")
    plt.plot(x_train, y_init[0], "o", label="Initial Model Prediction[0]")
    plt.plot(x_train, y_init[1], "o", label="Initial Model Prediction[1]")
    plt.plot(x_train, y_init[2], "o", label="Initial Model Prediction[2]")
    plt.plot(x_train, y_pred[0], "--", label="Final Model Prediction[0]")
    plt.plot(x_train, y_pred[1], "--", label="Final Model Prediction[1]")
    plt.plot(x_train, y_pred[2], "--", label="Final Model Prediction[2]")
    plt.legend()
    plt.savefig(
        f"./image/hamiltonian-{name}-{nqubit}-{depth}-{datetime.now().strftime('%Y%m%d%H%M')}"
    )
    # plt.show()


def create_y(nqubit, ansatz, time_step, xs, params):
    U_out = ansatz.create_ansatz(params)

    U_time = create_U_time(nqubit, ansatz, time_step)

    y_pred = np.array([qcl_pred(nqubit, x, U_time, U_out) for x in xs])

    return y_pred


if __name__ == "__main__":
    args = sys.argv
    path = args[1]
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    nqubit = config["nqubit"]

    ## creat train data
    x_train, y_train = create_train_data(nqubit)
    # create_graph(x_train, y_train.T)

    # create Unitary gate instance
    ansatz = create_ansatz(config)

    random_list, bounds = randomize(nqubit, config)
    y_init = create_y(nqubit, ansatz, time_step, x_train, random_list)

    result = minimize(cost, random_list, method="BFGS")
    # print(result)
    y_pred = create_y(nqubit, ansatz, time_step, x_train, result.x)

    create_graph(
        x_train,
        y_init.T,
        y_train.T,
        y_pred.T,
        ansatz.ansatz_type.name,
        config["nqubit"],
        config["depth"],
    )
