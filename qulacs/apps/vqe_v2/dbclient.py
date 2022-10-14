import sqlite3
from pathlib import Path


class DBClient:
    def __init__(self, filepath):
        self.conn = sqlite3.connect(filepath)

    def create_table(self, force=False):
        cur = self.conn.cursor()
        if force:
            cur.execute("DROP TABLE IF EXISTS jobs")

        sql = Path("sql/create_table_job.sql").read_text()
        cur.execute(sql)
        self.conn.commit()

    def insertJob(self, job):
        cur = self.conn.cursor()
        sql = Path("sql/insert_job.sql").read_text()
        cur.execute(
            sql,
            (
                job.id,
                job.creation_time,
                job.execution_second,
                job.nqubit,
                job.depth,
                job.gate_type,
                job.gate_set,
                job.bn_type,
                job.bn_range,
                job.bn,
                job.cn,
                job.r,
                job.t_type,
                job.max_time,
                job.min_time,
                job.t,
                job.cost,
                job.parameter,
                job.iteration,
                job.cost_history,
                job.parameter_history,
                job.iteration_history,
                job.noise_singlequbit_enabled,
                job.noise_singlequbit_value,
                job.noise_twoqubit_enabled,
                job.noise_twoqubit_value,
                job.config,
            ),
        )
        self.conn.commit()

    def find(self):
        sql = Path("sql/find_all_job.sql").read_text()
        cur = self.conn.cursor()
        cur.execute(sql)
        jobs = cur.fetchall()
        return jobs

    def findJobByTime(self, nqubit, depth, gate_type, t_type, t_min, t_max):
        cur = self.conn.cursor()
        sql = Path("sql/find_job_by_time.sql").read_text()
        cur.execute(sql, (nqubit, depth, gate_type, t_type, t_min, t_max))
        jobs = cur.fetchall()
        return jobs

    def findJob(
        self, nqubit, depth, gate_type, bn_type=None, bn_range=None, bn_value=None
    ):
        cur = self.conn.cursor()
        if bn_type == "static":
            sql = Path("sql/find_job_by_gate_type_and_bn_value.sql").read_text()
            cur.execute(sql, (nqubit, depth, gate_type, bn_value))
        elif bn_type == "static_random":
            sql = Path("sql/find_job_by_gate_type_and_bn_range.sql").read_text()
            cur.execute(sql, (nqubit, depth, gate_type, bn_range))
        else:
            sql = Path("sql/find_job_by_gate_type.sql").read_text()
            cur.execute(sql, (nqubit, depth, gate_type))
        jobs = cur.fetchall()
        return jobs
