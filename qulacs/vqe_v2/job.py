class Job:
  def __init__(self, creation_time, execution_second, nqubit, depth, gate_type, gate_set, bn_type, bn_range, bn, cn, r, t_type, max_time, min_time, t, cost, parameter, iteration, cost_history, parameter_history, iteration_history):
    self.creation_time = creation_time
    self.execution_second = execution_second
    self.nqubit = nqubit
    self.depth = depth
    self.gate_type = gate_type
    self.gate_set = gate_set
    self.bn_type = bn_type
    self.bn_range = bn_range
    self.bn = bn
    self.cn = cn
    self.r = r
    self.t_type = t_type
    self.max_time = max_time
    self.min_time = min_time
    self.t = t
    self.cost = cost
    self.parameter = parameter
    self.iteration = iteration
    self.cost_history = cost_history
    self.parameter_history = parameter_history
    self.iteration_history = iteration_history
