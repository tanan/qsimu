from collections.abc import Sequence
from typing import Any

from common.database.bigquery import BigQueryClient
from common.database.bigquery.sql.find_job import sql_for_find_job
from common.database.schema.job import Job
from google.cloud import bigquery

DATASET = "vqe"
TABLE = "job_result"


def create_job_result_table(client: BigQueryClient) -> None:
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("creation_time", "DATETIME"),
        bigquery.SchemaField("execution_second", "FLOAT64"),
        bigquery.SchemaField("nqubit", "INTEGER"),
        bigquery.SchemaField("depth", "INTEGER"),
        bigquery.SchemaField("gate_type", "STRING"),
        bigquery.SchemaField("gate_set", "STRING"),
        bigquery.SchemaField("bn_type", "STRING"),
        bigquery.SchemaField("bn_range", "INTEGER"),
        bigquery.SchemaField("bn", "STRING"),
        bigquery.SchemaField("cn", "STRING"),
        bigquery.SchemaField("r", "STRING"),
        bigquery.SchemaField("t_type", "STRING"),
        bigquery.SchemaField("min_time", "STRING"),
        bigquery.SchemaField("max_time", "STRING"),
        bigquery.SchemaField("t", "STRING"),
        bigquery.SchemaField("cost", "STRING"),
        bigquery.SchemaField("parameter", "STRING"),
        bigquery.SchemaField("iteration", "STRING"),
        bigquery.SchemaField("cost_history", "STRING"),
        bigquery.SchemaField("parameter_history", "STRING"),
        bigquery.SchemaField("iteration_history", "STRING"),
        bigquery.SchemaField("noise_singlequbit_enabled", "BOOL"),
        bigquery.SchemaField("noise_singlequbit_value", "STRING"),
        bigquery.SchemaField("noise_twoqubit_enabled", "BOOL"),
        bigquery.SchemaField("noise_twoqubit_value", "STRING"),
        bigquery.SchemaField("config", "STRING"),
    ]
    table = client.create_table(DATASET, TABLE, schema)
    print(
        "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
    )


def insert_job_result(client: BigQueryClient, job: Job) -> None:
    row = vars(job)  ## convert dict type
    row["creation_time"] = row["creation_time"].strftime(
        "%Y-%m-%d %H:%M:%S"
    )  ## convert to str from datetime
    errors = client.insert_rows(DATASET, TABLE, [row])
    if errors == []:
        print("New rows have been added.")
    # else:
    #     print("Encountered errors while inserting rows: {}".format(errors))


def find_job_result(
    client: BigQueryClient, filter: str = None
) -> Sequence[dict[str, Any]]:
    """
    Find job results of vqe expectation.

    Params are configured following values.

        client: A client to connect and operate BigQuery.
        filter: sql phrase to filter records. It excludes `filter`.
    """
    if filter is None:
        return client.client.query(sql_for_find_job())
    else:
        return client.client.query("{} WHERE {}".format(sql_for_find_job(), filter))
