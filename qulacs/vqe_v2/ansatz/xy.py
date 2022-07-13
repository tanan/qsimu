import sys
import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import CNOT, RY, RZ, merge

sys.path.append('..')
from ansatz.ansatz import Ansatz
from ansatz.pauli_gate import PauliGate

class XYAnsatz(Ansatz):

  def __init__(self, nqubit, depth, gate_set, time, bn, gamma=0):
    super().__init__(nqubit, depth, gate_set, time, bn, gamma)

  def create_hamiltonian(self, cn, bn, gamma):
    XX= np.array(np.zeros(2**self.nqubit))
    YY= np.array(np.zeros(2**self.nqubit))
    Zn= np.array(np.zeros(2**self.nqubit))
    for k in range(self.nqubit-1):
      for l in range(self.nqubit):
        if k==l:
          if l==0:
            hamiX = PauliGate.X_gate.value
            hamiY = PauliGate.Y_gate.value
          else:
            hamiX = np.kron(hamiX,PauliGate.X_gate.value)
            hamiY = np.kron(hamiY,PauliGate.Y_gate.value)

        elif k+1==l:
          hamiX = np.kron(hamiX,PauliGate.X_gate.value)
          hamiY = np.kron(hamiY,PauliGate.Y_gate.value)
        else:
          if l==0:
            hamiX = PauliGate.I_gate.value
            hamiY = PauliGate.I_gate.value
          else:
            hamiX = np.kron(hamiX,PauliGate.I_gate.value)
            hamiY = np.kron(hamiY,PauliGate.I_gate.value)
      XX = XX+ 0.5*cn[k]*(1+gamma)*hamiX
      YY = YY+ 0.5*cn[k]*(1-gamma)*hamiY

    for m in range(self.nqubit):
      for n in range(self.nqubit):
        if m==n:
          if n==0:
            hamiZ = PauliGate.Z_gate.value
          else:
            hamiZ = np.kron(hamiZ,PauliGate.Z_gate.value)

        else:
          if n==0:
            hamiZ = PauliGate.I_gate.value
          else:
            hamiZ = np.kron(hamiZ,PauliGate.I_gate.value)

      Zn = Zn + bn['value'][m]*hamiZ

    hamiltonian = XX + YY + Zn
    return np.linalg.eigh(hamiltonian)

  def create_ansatz(self, random_list):
    '''
    ansatzを作成する
    ハミルトニアンゲートはdepthごとに別々のパラメータとなる.
    bnについてはdepth x nqubitの配列をパラメータとして渡す.
    TODO: r, cnのrandomizeは未実装
    '''
    circuit = QuantumCircuit(self.nqubit)
    for d in range(self.depth):
      circuit.add_gate(CNOT(0, 1))
      if self.bn['type'] == "random":
        circuit.add_gate(merge(RY(0, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)]), RZ(0, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)+1])))
        circuit.add_gate(merge(RY(1, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)+2]), RZ(1, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)+3])))
        circuit.add_gate(self.create_hamiltonian_gate([1]*self.nqubit, 0, random_list[d+self.depth:d+self.depth+self.nqubit], random_list[d]))
      elif self.bn['type'] == "static" or self.bn['type'] == "static_random":
        circuit.add_gate(merge(RY(0, random_list[self.depth+(self.gate_set*d)]), RZ(0, random_list[self.depth+(self.gate_set*d)+1])))
        circuit.add_gate(merge(RY(1, random_list[self.depth+(self.gate_set*d)+2]), RZ(1, random_list[self.depth+(self.gate_set*d)+3])))
        if self.time['type'] == "random":
          circuit.add_gate(self.create_hamiltonian_gate(random_list[d]))
        else:
          circuit.add_gate(self.create_hamiltonian_gate(self.time['min_val']))

    return circuit
