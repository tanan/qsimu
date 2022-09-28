import numpy as np
from scipy.optimize import Bounds


def generate_random_time(min_time, max_time, num):
    return np.random.uniform(min_time, max_time, num)


def generate_random_bn(min_bn, max_bn, num):
    return np.random.uniform(-1.0, 1.0, num)


def randomize(nqubit, config):
    """
    Create random list for a parametric ciruit.
    Parameters are time, cn, r(gamma), bn
    time: 0 - max_time
    cn: 0 - 1
    r(gamma): 0 - 1
    bn: 0 - 1
    theta: 0 - 1
    Return [t1, t2, ... td, cn1, cn2, ... cnd, r1, r2, ..., rd, bn1, bn2, ..., bnd, theta1, ... theatad*gate_set]
    """
    if config["gate"]["type"] == "direct":
        list_count = 2 * nqubit * (config["depth"] + 1)
        return (
            np.random.random(list_count) * 1e-1,
            Bounds([-np.Inf] * list_count, [np.Inf] * list_count),
        )

    ## init time
    init_random_list = np.array([])
    if config["gate"]["time"]["type"] == "random":
        init_random_list = generate_random_time(
            config["gate"]["time"]["min_val"],
            config["gate"]["time"]["max_val"],
            config["depth"],
        )

    ## init bn if type is random
    if config["gate"]["bn"]["type"] == "random" and config["gate"]["type"] not in [
        "indirect_xyz",
        "direct",
    ]:
        init_random_list = np.append(
            init_random_list, generate_random_bn(-1.0, 1.0, config["depth"] * nqubit)
        )

    init_random_list = np.append(
        init_random_list,
        np.random.random(
            config["gate"]["parametric_rotation_gate_set"] * config["depth"]
        )
        * 1e-1,
    )
    if config["gate"]["bounds"]:
        return init_random_list, get_bounds(nqubit, config)

    return init_random_list, None


def get_bounds(nqubit, config):
    t_min = np.array([config["gate"]["time"]["min_val"]] * config["depth"])
    t_max = np.array([config["gate"]["time"]["max_val"]] * config["depth"])
    bn_min = np.array([0.0] * config["depth"] * nqubit)
    bn_max = np.array([1.0] * config["depth"] * nqubit)
    theta_min = np.array(
        [-np.Inf] * (config["gate"]["parametric_rotation_gate_set"] * config["depth"])
    )
    theta_max = np.array(
        [np.Inf] * (config["gate"]["parametric_rotation_gate_set"] * config["depth"])
    )

    if (
        config["gate"]["bn"]["type"] == "random"
        and config["gate"]["type"] != "indirect_xyz"
    ):
        min_bounds = np.append(
            np.append(np.append(np.array([]), t_min), bn_min), theta_min
        )
        max_bounds = np.append(
            np.append(np.append(np.array([]), t_max), bn_max), theta_max
        )
    else:
        min_bounds = np.append(np.append(np.array([]), t_min), theta_min)
        max_bounds = np.append(np.append(np.array([]), t_max), theta_max)

    return Bounds(min_bounds, max_bounds)
