from google.cloud import bigquery
from common.database.bigquery import BigQueryClient
from common.database.schema.job import Job

dataset = "vqe"
table = "job_result"


def create_job_result_table(client: BigQueryClient) -> None:
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("creation_time", "DATETIME"),
        bigquery.SchemaField("execution_second", "INTEGER"),
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
        bigquery.SchemaField("noise_singlequbit_enabled", "STRING"),
        bigquery.SchemaField("noise_singlequbit_value", "STRING"),
        bigquery.SchemaField("noise_twoqubit_enabled", "STRING"),
        bigquery.SchemaField("noise_twoqubit_value", "STRING"),
        bigquery.SchemaField("config", "STRING"),
    ]
    table = client.create_table(dataset, table, schema)
    print(
        "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
    )


def insert_job_result(client: BigQueryClient, job: Job) -> None:
    errors = client.insert_rows(dataset, table, [vars(job)])
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))
