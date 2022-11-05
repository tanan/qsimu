from collections.abc import Sequence
from typing import Any

import numpy as np
from common.database.bigquery import BigQueryClient
from common.database.bigquery.job_result import find_job_result


def get_result_from_bq(project_id: str, where: str = None) -> Sequence[dict[str, Any]]:
    return find_job_result(BigQueryClient(project_id), where)


def _distinct_dict_value(jobs: Sequence[dict[str, Any]]) -> dict[str, iter]:
    config = {}
    config["nqubit"] = set(job["nqubit"] for job in jobs)
    config["gate_type"] = set(job["gate_type"] for job in jobs)
    config["depth"] = set(job["depth"] for job in jobs)
    config["t_type"] = set(job["t_type"] for job in jobs)
    config["min_time"] = set(job["min_time"] for job in jobs)
    config["max_time"] = set(job["max_time"] for job in jobs)
    config["bn_type"] = set(job["bn_type"] for job in jobs)
    config["bn_range"] = set(job["bn_range"] for job in jobs)
    config["bn"] = set(job["bn"] for job in jobs)
    config["noise_singlequbit_value"] = set(
        job["noise_singlequbit_value"] for job in jobs
    )
    config["noise_twoqubit_value"] = set(job["noise_twoqubit_value"] for job in jobs)
    config["constraints"] = set(job["constraints"] for job in jobs)
    config["bounds"] = set(job["bounds"] for job in jobs)
    return config


def _generate_all_config_combinations_for_time_summary(
    config: dict[str, iter]
) -> Sequence[dict[str, Any]]:
    combinations = []
    for nqubit in config["nqubit"]:
        for gate_type in config["gate_type"]:
            for depth in config["depth"]:
                for t_type in config["t_type"]:
                    for min_time in config["min_time"]:
                        for max_time in config["max_time"]:
                            for noise_singlequbit_value in config[
                                "noise_singlequbit_value"
                            ]:
                                for noise_twoqubit_value in config[
                                    "noise_twoqubit_value"
                                ]:
                                    for constraints in config["constraints"]:
                                        for bounds in config["bounds"]:

                                            val = {}
                                            val["nqubit"] = nqubit
                                            val["gate_type"] = gate_type
                                            val["depth"] = depth
                                            val["t_type"] = t_type
                                            val["min_time"] = min_time
                                            val["max_time"] = max_time
                                            val[
                                                "noise_singlequbit_value"
                                            ] = noise_singlequbit_value
                                            val[
                                                "noise_twoqubit_value"
                                            ] = noise_twoqubit_value
                                            val["constraints"] = constraints
                                            val["bounds"] = bounds
                                            combinations.append(val)
    return combinations


def _filter_job(
    jobs: Sequence[dict[str, Any]], config: dict[str, Any]
) -> Sequence[dict[str, Any]]:
    return list(
        filter(
            lambda x: x["bounds"] == config["bounds"],
            filter(
                lambda x: x["constraints"] == config["constraints"],
                filter(
                    lambda x: x["noise_twoqubit_value"]
                    == config["noise_twoqubit_value"],
                    filter(
                        lambda x: x["noise_singlequbit_value"]
                        == config["noise_singlequbit_value"],
                        filter(
                            lambda x: x["max_time"] == config["max_time"],
                            filter(
                                lambda x: x["min_time"] == config["min_time"],
                                filter(
                                    lambda x: x["t_type"] == config["t_type"],
                                    filter(
                                        lambda x: x["depth"] == config["depth"],
                                        filter(
                                            lambda x: x["gate_type"]
                                            == config["gate_type"],
                                            filter(
                                                lambda x: x["nqubit"]
                                                == config["nqubit"],
                                                jobs,
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def _convert_result_by_time_to_dict(config, cost_list, iter_list):
    return {
        "nqubit": config["nqubit"],
        "gate_type": config["gate_type"],
        "depth": config["depth"],
        "t_type": config["t_type"],
        "min_time": config["min_time"],
        "max_time": config["max_time"],
        "noise_singlequbit_value": config["noise_singlequbit_value"],
        "noise_twoqubit_value": config["noise_twoqubit_value"],
        "constraints": config["constraints"],
        "bounds": config["bounds"],
        "cost": {
            "min": np.min(cost_list),
            "max": np.max(cost_list),
            "mean": np.mean(cost_list),
            "std": np.std(cost_list),
        },
        "iter": {
            "min": np.min(iter_list),
            "max": np.max(iter_list),
            "mean": np.mean(iter_list),
            "std": np.std(iter_list),
        },
    }


def summary_job_result_by_time(
    jobs: Sequence[dict[str, Any]]
) -> Sequence[dict[str, Any]]:
    if len(jobs) == 0:
        return None

    combinations = _generate_all_config_combinations_for_time_summary(
        _distinct_dict_value(jobs)
    )

    summary = []

    for config in combinations:
        filter_jobs = _filter_job(jobs, config)

        if len(filter_jobs) == 0:
            continue

        cost_list = []
        iter_list = []

        for job in filter_jobs:
            cost_list = np.append(cost_list, float(job["cost"]))
            iter_list = np.append(iter_list, float(job["iteration"]))

        summary.append(_convert_result_by_time_to_dict(config, cost_list, iter_list))

    return summary


# def getResult(
#     jobs: Sequence[dict[str, Any]],
#     nqubit: int,
#     depth: int,
#     method: str,
#     bn_type: str = None,
#     bn_range: int = None,
#     bn_value: str = None,
# ):
#     ## init numpy array
#     cost_list = np.array([])
#     iter_list = np.array([])

#     if len(jobs) == 0:
#         return None

#     for job in jobs:
#         cost_list = np.append(cost_list, float(job["cost"]))
#         iter_list = np.append(iter_list, float(job["iter"]))

#     return {
#         "method": method,
#         "depth": depth,
#         "bn_type": bn_type,
#         "bn_range": bn_range,
#         "bn_value": str(bn_value).strip("[").strip("]").split(",")[0],
#         "cost": {
#             "min": np.min(cost_list),
#             "max": np.max(cost_list),
#             "mean": np.mean(cost_list),
#             "std": np.std(cost_list),
#         },
#         "iter": {
#             "min": np.min(iter_list),
#             "max": np.max(iter_list),
#             "mean": np.mean(iter_list),
#             "std": np.std(iter_list),
#         },
#     }
