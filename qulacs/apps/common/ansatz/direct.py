import sys
import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge

sys.path.append('..')
from .ansatz import Ansatz

class DirectAnsatz(Ansatz):
  def __init__(self, nqubit, depth, gate_set):
    super().__init__(nqubit, depth, gate_set)

  def create_hamiltonian(self, cn=None, bn=None):
    return None, None

  def create_ansatz(self, random_list):
    circuit = QuantumCircuit(self.nqubit)
    for d in range(self.depth):
      for i in range(self.nqubit):
        circuit.add_gate(merge(RY(i, random_list[2*i+2*self.nqubit*d]), RZ(i, random_list[2*i+1+2*self.nqubit*d])))
      for i in range(self.nqubit//2):
        circuit.add_gate(CZ(2*i, 2*i+1))
      for i in range(self.nqubit//2-1):
        circuit.add_gate(CZ(2*i+1, 2*i+2))
    for i in range(self.nqubit):
      circuit.add_gate(merge(RY(i, random_list[2*i+2*self.nqubit*self.depth]), RZ(i, random_list[2*i+1+2*self.nqubit*self.depth])))

    return circuit
