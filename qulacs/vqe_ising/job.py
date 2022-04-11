class Job:
  def __init__(self, creation_time, execution_second, nqubit, depth, gate_type, gate_set, bn, cn, r, max_time, cost, parameter, iteration):
    self.creation_time = creation_time
    self.execution_second = execution_second
    self.nqubit = nqubit
    self.depth = depth
    self.gate_type = gate_type
    self.gate_set = gate_set
    self.bn = bn
    self.cn = cn
    self.r = r
    self.max_time = max_time
    self.cost = cost
    self.parameter = parameter
    self.iteration = iteration