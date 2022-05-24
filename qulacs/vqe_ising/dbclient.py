import sqlite3

class DBClient:
  def __init__(self, filepath):
    self.conn = sqlite3.connect(filepath)
  
  def create_table(self, force=False):
    cur = self.conn.cursor()
    if force:
      cur.execute("DROP TABLE IF EXISTS jobs")

    cur.execute(
      """
      CREATE TABLE jobs(
        id INTEGER PRIMARY KEY,
        creation_time TIMESTAMP,
        execution_second INTEGER,
        nqubit INTEGER,
        depth INTEGER,
        gate_type TEXT,
        gate_set TEXT,
        bn_type TEXT,
        bn TEXT,
        cn TEXT,
        r TEXT,
        max_time TEXT,
        cost TEXT,
        parameter TEXT,
        iteration TEXT,
        cost_history TEXT,
        parameter_history TEXT,
        iteration_history TEXT
      )
      """
    )
    self.conn.commit()

  def insert(self, job):
    cur = self.conn.cursor()
    cur.execute(
    """
      INSERT INTO jobs (
        creation_time,
        execution_second,
        nqubit,
        depth,
        gate_type,
        gate_set,
        bn_type,
        bn,
        cn,
        r,
        max_time,
        cost,
        parameter,
        iteration,
        cost_history,
        parameter_history,
        iteration_history
      ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    ,(
      job.creation_time,
      job.execution_second,
      job.nqubit,
      job.depth,
      job.gate_type,
      job.gate_set,
      job.bn_type,
      job.bn,
      job.cn,
      job.r,
      job.max_time,
      job.cost,
      job.parameter,
      job.iteration,
      job.cost_history,
      job.parameter_history,
      job.iteration_history
    ))
    self.conn.commit()

  def select(self):
    cur = self.conn.cursor()
    cur.execute(
      """
        SELECT
          id,
          creation_time,
          execution_second,
          nqubit,
          depth,
          gate_type,
          gate_set,
          bn_type,
          bn,
          cn,
          r,
          max_time,
          cost,
          parameter,
          iteration,
          cost_history,
          parameter_history,
          iteration_history
        FROM jobs
      """
    )
    jobs = cur.fetchall()
    for job in jobs:
      print(job)

  def selectResultsByNQubitAndGateType(self, nqubit, gate_type, bn, depth):
    cur = self.conn.cursor()
    cur.execute(
      """
        SELECT
          id,
          creation_time,
          execution_second,
          nqubit,
          depth,
          gate_type,
          gate_set,
          bn_type,
          bn,
          cn,
          r,
          max_time,
          cost,
          parameter,
          iteration,
          cost_history,
          parameter_history,
          iteration_history
        FROM jobs
        WHERE
          nqubit = ?
        AND
          gate_type = ?
        AND
          bn = ?
        AND
          depth = ?
      """
    ,(
      nqubit,
      gate_type,
      bn,
      depth
    ))
    jobs = cur.fetchall()
    return jobs

  def selectResultsByNQubitAndGateTypeAndBnType(self, nqubit, gate_type, bn_type, depth):
    cur = self.conn.cursor()
    cur.execute(
      """
        SELECT
          id,
          creation_time,
          execution_second,
          nqubit,
          depth,
          gate_type,
          gate_set,
          bn_type,
          bn,
          cn,
          r,
          max_time,
          cost,
          parameter,
          iteration,
          cost_history,
          parameter_history,
          iteration_history
        FROM jobs
        WHERE
          nqubit = ?
        AND
          gate_type = ?
        AND
          bn_type = ?
        AND
          depth = ?
      """
    ,(
      nqubit,
      gate_type,
      bn_type,
      depth
    ))
    jobs = cur.fetchall()
    return jobs