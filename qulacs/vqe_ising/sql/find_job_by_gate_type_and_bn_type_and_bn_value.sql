SELECT
  id,
  creation_time,
  execution_second,
  nqubit,
  depth,
  gate_type,
  gate_set,
  bn_type,
  bn_range,
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
  depth = ?
AND
  gate_type = ?
AND
  bn_type = ?
AND
  bn = ?
