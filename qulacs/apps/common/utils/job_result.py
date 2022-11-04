import numpy as np
from typing import Any
from collections.abc import Sequence
from common.database.bigquery import BigQueryClient
from common.database.bigquery.job_result import find_job_result


def get_result_from_bq(project_id: str, where: str = None) -> Sequence[dict[str, Any]]:
    client = BigQueryClient(project_id)
    return find_job_result(client, where)


def summary_job_result_by_time(
    jobs: Sequence[dict[str, Any]],
    nqubit: int,
    method: str,
    depth: int,
    t_type: str,
    t_min: int,
    t_max: int,
):
    if len(jobs) == 0:
        return None

    for job in jobs:
        cost_list = np.append(cost_list, float(job[cost_num]))
        iter_list = np.append(iter_list, float(job[iter_num]))

    return {
        "nqubit": nqubit,
        "method": method,
        "depth": depth,
        "t_type": t_type,
        "t_min": t_min,
        "t_max": t_max,
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
