import numpy as np
from dbclient import DBClient

cost_num = 13
iter_num = 15


def getResult(nqubit, depth, method, bn_type=None, bn_range=None, bn_value=None):
    ## init numpy array
    cost_list = np.array([])
    iter_list = np.array([])

    ## db client
    db = DBClient("data/job_results.sqlite3")
    jobs = db.findJob(nqubit, depth, method, bn_type, bn_range, bn_value)

    if len(jobs) == 0:
        return None

    for job in jobs:
        cost_list = np.append(cost_list, float(job[cost_num]))
        iter_list = np.append(iter_list, float(job[iter_num]))

    return {
        "method": method,
        "depth": depth,
        "bn_type": bn_type,
        "bn_range": bn_range,
        "bn_value": str(bn_value).strip("[").strip("]").split(",")[0],
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
