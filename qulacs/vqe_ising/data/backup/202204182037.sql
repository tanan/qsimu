PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
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
      );
COMMIT;
