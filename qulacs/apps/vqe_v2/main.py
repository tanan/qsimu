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

    return hamiltonian.get_expectation_value(state)


def grad_fn(nqubit, ansatz, hamiltonian, params):
    delta = 1e-6
    v = []

    for i in range(len(params)):
        a = list(params)
        a[i] = params[i] + delta
        v.append(a)
        a = list(params)
        a[i] = params[i] - delta
        v.append(a)

    estimates = []
    for p in v:
        estimates.append(estimator(nqubit, ansatz, hamiltonian, p))

    grad = []
    for i in range(len(params)):
        d = estimates[2 * i] - estimates[2 * i + 1]
        grad.append(d / (2 * delta))

    return np.array(grad)


def record(x):
    global param_history
    global cost_history
    global iter_history
    param_history.append(x)
    cost_history.append(cost(x))
    iter_history.append(iteration)


def vqe(nqubit, ansatz, hamiltonian, init_params, cost_fn, grad_fn, optimizer):
    def g_fn(params):
        return grad_fn(nqubit, ansatz, hamiltonian, params)

    opt_state = optimizer.get_init_state(init_params)
    while True:
        opt_state = optimizer.step(opt_state, cost_fn, g_fn)
        print(f"[Iteration: {opt_state.niter}]  [Cost: {opt_state.cost}]")
        print(f"{opt_state.status}")
        # print(opt_state)
        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state


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
    qulacs_hamiltonian = create_ising_hamiltonian(config["nqubit"])

    ## init ansatz instance
    global ansatz
    ansatz = init_ansatz(config)

    ## randomize and create constraints
    init_random_list, bounds = randomize(config["nqubit"], config)
    record(init_random_list)

    ## calculation
    options = {"maxiter": 1000}
    if config["gate"]["constraints"]:
        constraints = create_time_constraints(config["depth"], len(init_random_list))
        opt = minimize(
            cost,
            init_random_list,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options=options,
            callback=record,
        )
    elif config["gate"]["bounds"]:
        opt = minimize(
            cost,
            init_random_list,
            method="SLSQP",
            options=options,
            bounds=bounds,
            callback=record,
        )
    else:
        # def c_fn(params):
        #     return cost_fn(config["nqubit"], ansatz, qulacs_hamiltonian, params)

        # def g_fn(params):
        #     return grad_fn(config["nqubit"], ansatz, qulacs_hamiltonian, params)

        # global optimizer
        # result = vqe(config["nqubit"], ansatz, qulacs_hamiltonian, init_random_list, cost, grad_fn, optimizer)
        # print(result)
        # return
        # opt = minimize(
        #     cost, init_random_list, jac=g_fn ,method="BFGS", options=options, callback=record
        # )
        opt = minimize(
            cost, init_random_list, method="BFGS", options=options, callback=record
        )

    end_time = time.perf_counter()

    ## record to database
    job = JobFactory(config).create(
        now, start_time, end_time, cost_history, param_history, iter_history
    )
    client = DBClient("data/job_results.sqlite3")
    sqlitejob.insert_job(client, job)
    if config["gcp"]["bigquery"]["import"]:
        bqClient = BigQueryClient(config["gcp"]["project"]["id"])
        insert_job_result(bqClient, job)
    # output(param_history, cost_history, iter_history)
    # np.savetxt('data/xy_params.txt', param_history[-1])
    img = circuit_drawer(ansatz.create_ansatz(init_random_list), "latex")
    img.save(f"image/{job.id}.png")


def start(config):
    reset()
    run(config)


if __name__ == "__main__":
    args = sys.argv
    path = args[1]
    iter_for_static = 10
    iter_for_random = 100
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        if (
            config["gate"]["type"] == ["direct", "indirect_xyz"]
            or config["gate"]["bn"]["type"] == "static"
        ):
            for k in range(iter_for_static):
                start(config)
        else:
            for k in range(iter_for_random):
                config["gate"]["bn"]["value"] = np.random.rand(
                    config["nqubit"]
                ) * config["gate"]["bn"]["range"] - (config["gate"]["bn"]["range"] / 2)
                start(config)
