import sqlite3
from pathlib import Path


class DBClient:
    def __init__(self, filepath):
        self.conn = sqlite3.connect(filepath)

    def find(self):
        """
        TODO: move this function to common module.
        """
        sql = Path("sql/find_all_job.sql").read_text()
        cur = self.conn.cursor()
        cur.execute(sql)
        jobs = cur.fetchall()
        return jobs

    def findJobByTime(self, nqubit, depth, gate_type, t_type, t_min, t_max):
        """
        TODO: move this function to common module.
        """
        cur = self.conn.cursor()
        sql = Path("sql/find_job_by_time.sql").read_text()
        cur.execute(sql, (nqubit, depth, gate_type, t_type, t_min, t_max))
        jobs = cur.fetchall()
        return jobs

    def findJob(
        self, nqubit, depth, gate_type, bn_type=None, bn_range=None, bn_value=None
    ):
        """
        TODO: move this function to common module.
        """
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
