from job import Job
from utils import to_string

class JobFactory():
  def __init__(self, config):
    self.config = config
    pass
  
  def create(self, current_time, start_time, end_time, cost_history, param_history, iter_history):
    if self.config['gate']['type'] == 'direct':
      job = Job(
        current_time,
        end_time - start_time,
        self.config['nqubit'],
        self.config['depth'],
        self.config['gate']['type'],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        str(cost_history[-1]),
        str(param_history[-1]),
        str(iter_history[-1]),
        str(cost_history),
        to_string(param_history),
        str(iter_history)
      )
    else:
      job = Job(
        current_time,
        end_time - start_time,
        self.config['nqubit'],
        self.config['depth'],
        self.config['gate']['type'],
        str(self.config['gate']['parametric_rotation_gate_set']),
        str(self.config['gate']['bn']['type']),
        self.config['gate']['bn']['range'] if 'range' in self.config['gate']['bn'] else None,
        str(self.config['gate']['bn']['value']),
        str(self.config['gate']['cn']['value']),
        str(self.config['gate']['r']['value']),
        self.config['gate']['time']['type'],
        self.config['gate']['time']['max_val'],
        self.config['gate']['time']['min_val'],
        str(self.config['gate']['time']['value']),
        str(cost_history[-1]),
        str(param_history[-1]),
        str(iter_history[-1]),
        str(cost_history),
        to_string(param_history),
        str(iter_history)
      )
    return job