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
  t_type,
  max_time,
  min_time,
  t,
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
  t_type = ?
AND
  min_time = ?
AND
  max_time = ?