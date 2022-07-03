import sys
import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge

sys.path.append('..')
from ansatz.ansatz import Ansatz
from hamiltonian import create_hamiltonian_gate

class DirectAnsatz(Ansatz):
  def __init__(self, nqubit, depth, gate_set):
    super().__init__(nqubit, depth, gate_set)

  def create_hamiltonian(self, cn=None, bn=None):
    return None, None

  def create_ansatz(self, random_list):
    circuit = QuantumCircuit(self.nqubit)
    circuit.add_gate(create_hamiltonian_gate(self.nqubit, 0.77))
    for d in range(self.nqubit):
      circuit.add_RX_gate(d, random_list[(self.gate_set*d)])
      circuit.add_RZ_gate(d, random_list[(self.gate_set*d)+1])
      circuit.add_RX_gate(d, random_list[(self.gate_set*d)+2])
    return circuit