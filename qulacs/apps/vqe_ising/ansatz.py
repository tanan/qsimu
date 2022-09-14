from mimetypes import init
import qulacs
from qulacs import QuantumCircuit
from qulacs.gate import CNOT, CZ, RX, RY, RZ, merge, DenseMatrix
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm

class AnsatzDirect:
  def __init__(self, nqubit, depth):
    self.nqubit = nqubit
    self.depth = depth

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

class AnsatzIndirectByIsing:

  def __init__(self, nqubit, depth, gate_set, bn):
    self.nqubit = nqubit
    self.depth = depth
    self.gate_set = gate_set
    self.bn = bn
    self.diag, self.eigen_vecs = self.create_hamiltonian(bn['value'], [1]*self.nqubit)

  def create_hamiltonian(self, bn, cn):
    I_gate = [[1,0],[0,1]]
    X_gate = [[0,1],[1,0]]
    Y_gate = [[0,0-1j],[0+1j,0]]
    
    XX= np.array(np.zeros(2**self.nqubit))
    Y= np.array(np.zeros(2**self.nqubit))
    for k in range(self.nqubit-1):
        for l in range(self.nqubit):
            if k==l:
                if l==0:
                    hamiX = X_gate
                else:
                    hamiX = np.kron(hamiX,X_gate)

            elif k+1==l:
                hamiX = np.kron(hamiX,X_gate)
            else:
                if l==0:
                    hamiX = I_gate
                else:
                    hamiX = np.kron(hamiX,I_gate)
        XX = XX+ 0.5*cn[k]*hamiX

    for m in range(self.nqubit):
      for n in range(self.nqubit):
        if m==n:
          if n==0:
            hamiY = Y_gate
          else:
            hamiY = np.kron(hamiY,Y_gate)
        
        else:
          if n==0:
            hamiY = I_gate
          else:
            hamiY = np.kron(hamiY,I_gate)

      Y = Y + bn[m]*hamiY

    hamiltonian = XX + Y
    return np.linalg.eigh(hamiltonian)

  def create_hamiltonian_gate(self, t):
    time_evol_op = np.dot(np.dot(self.eigen_vecs, np.diag(np.exp(-1j*t*self.diag))), self.eigen_vecs.T.conj())
    return DenseMatrix([i for i in range(self.nqubit)], time_evol_op)

  def create_ansatz(self, random_list):
    '''
    ansatzを作成する
    ハミルトニアンゲートはdepthごとに別々のパラメータとなる.
    bnについてはdepth x nqubitの配列をパラメータとして渡す.
    TODO: r, cnのrandomizeは未実装
    '''
    circuit = QuantumCircuit(self.nqubit)
    for d in range(self.depth):
      if self.bn['type'] == "random":
        # + time + bn
        # circuit.add_gate(merge(RX(0, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)]), RY(0, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)+1])))
        # circuit.add_gate(merge(RX(1, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)+2]), RY(1, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)+3])))
        circuit.add_gate(RZ(0, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)]))
        circuit.add_gate(RZ(1, random_list[self.depth+(self.depth*self.nqubit)+(self.gate_set*d)+1]))
        circuit.add_gate(self.create_hamiltonian_gate(random_list[d+self.depth:d+self.depth+self.nqubit], random_list[d]))
      elif self.bn['type'] == "static" or self.bn['type'] == "static_random":
        # + time
        # circuit.add_gate(merge(RX(0, random_list[self.depth+(self.gate_set*d)]), RY(0, random_list[self.depth+(self.gate_set*d)+1])))
        # circuit.add_gate(merge(RX(1, random_list[self.depth+(self.gate_set*d)+2]), RY(1, random_list[self.depth+(self.gate_set*d)+3])))
        circuit.add_gate(RZ(0, random_list[self.depth+(self.gate_set*d)]))
        circuit.add_gate(RZ(1, random_list[self.depth+(self.gate_set*d)+1]))
        circuit.add_gate(self.create_hamiltonian_gate(random_list[d]))

    return circuit


class AnsatzIndirectByXYZ:

  def __init__(self, nqubit, depth, gate_set):
    self.nqubit = nqubit
    self.depth = depth
    self.gate_set = gate_set
    self.diag, self.eigen_vecs = self.create_hamiltonian([1]*self.nqubit)

  def create_hamiltonian(self, cn):
    I_gate = [[1,0],[0,1]]
    X_gate = [[0,1],[1,0]]
    Y_gate = [[0,0-1j],[0+1j,0]]
    Z_gate = [[1,0],[0,-1]]
    
    XX= np.array(np.zeros(2**self.nqubit))
    YY= np.array(np.zeros(2**self.nqubit))
    ZZ= np.array(np.zeros(2**self.nqubit))
    for k in range(self.nqubit-1):
        for l in range(self.nqubit):
            if k==l:
                if l==0:
                    hamiX = X_gate
                    hamiY = Y_gate
                    hamiZ = Z_gate
                else:
                    hamiX = np.kron(hamiX,X_gate)
                    hamiY = np.kron(hamiY,Y_gate)
                    hamiZ = np.kron(hamiZ,Z_gate)

            elif k+1==l:
                hamiX = np.kron(hamiX,X_gate)
                hamiY = np.kron(hamiY,Y_gate)
                hamiZ = np.kron(hamiZ,Z_gate)
            else:
                if l==0:
                    hamiX = I_gate
                    hamiY = I_gate
                    hamiZ = I_gate
                else:
                    hamiX = np.kron(hamiX,I_gate)
                    hamiY = np.kron(hamiY,I_gate)
                    hamiZ = np.kron(hamiZ,I_gate)
        XX = XX+ 0.5*cn[k]*hamiX
        YY = YY+ 0.5*cn[k]*hamiY
        ZZ = ZZ+ 0.5*cn[k]*hamiZ

    hamiltonian = XX + YY + ZZ
    return np.linalg.eigh(hamiltonian)

  def create_hamiltonian_gate(self, t):
    time_evol_op = np.dot(np.dot(self.eigen_vecs, np.diag(np.exp(-1j*t*self.diag))), self.eigen_vecs.T.conj())
    return DenseMatrix([i for i in range(self.nqubit)], time_evol_op)

  def create_ansatz(self, random_list):
    circuit = QuantumCircuit(self.nqubit)
    for d in range(self.depth):
        circuit.add_gate(merge(RX(0, random_list[self.depth+(self.gate_set*d)]), RY(0, random_list[self.depth+(self.gate_set*d)+1])))
        # TODO: cnをrandom化する
        circuit.add_gate(self.create_hamiltonian_gate(random_list[d]))

    return circuit

class AnsatzIndirectByXY:
  def __init__(self, nqubit, depth, gate_set, bn):
    self.nqubit = nqubit
    self.depth = depth
    self.gate_set = gate_set
    self.bn = bn
    self.diag, self.eigen_vecs = self.create_hamiltonian([1]*self.nqubit, 0, bn['value'])

  def create_hamiltonian(self, cn, gamma, bn):
    I_gate = [[1,0],[0,1]]
    X_gate = [[0,1],[1,0]]
    Y_gate = [[0,0-1j],[0+1j,0]]
    Z_gate = [[1,0],[0,-1]]
    
    XX= np.array(np.zeros(2**self.nqubit))
    YY= np.array(np.zeros(2**self.nqubit))
    Zn= np.array(np.zeros(2**self.nqubit))
    for k in range(self.nqubit-1):
        for l in range(self.nqubit):
            if k==l:
                if l==0:
                    hamiX = X_gate
                    hamiY = Y_gate
                else:
                    hamiX = np.kron(hamiX,X_gate)
                    hamiY = np.kron(hamiY,Y_gate)

            elif k+1==l:
                hamiX = np.kron(hamiX,X_gate)
                hamiY = np.kron(hamiY,Y_gate)
            else:
                if l==0:
                    hamiX = I_gate
                    hamiY = I_gate
                else:
                    hamiX = np.kron(hamiX,I_gate)
                    hamiY = np.kron(hamiY,I_gate)
        XX = XX+ 0.5*cn[k]*(1+gamma)*hamiX
        YY = YY+ 0.5*cn[k]*(1-gamma)*hamiY
        
    for m in range(self.nqubit):
        for n in range(self.nqubit):
            if m==n:
                if n==0:
                    hamiZ = Z_gate
                else:
                    hamiZ = np.kron(hamiZ,Z_gate)
            
            else:
                if n==0:
                    hamiZ = I_gate
                else:
                    hamiZ = np.kron(hamiZ,I_gate)
        
        Zn = Zn + bn[m]*hamiZ

    hamiltonian = XX + YY + Zn
    return np.linalg.eigh(hamiltonian)

  def create_hamiltonian_gate(self, t):
    time_evol_op = np.dot(np.dot(self.eigen_vecs, np.diag(np.exp(-1j*t*self.diag))), self.eigen_vecs.T.conj())
    return DenseMatrix([i for i in range(self.nqubit)], time_evol_op)

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
          circuit.add_gate(self.create_hamiltonian_gate(random_list[d]))

    return circuit

class AnsatzIndirectByNone:
  def __init__(self, nqubit, gate_set, depth):
    self.nqubit = nqubit
    self.depth = depth
    self.gate_set = gate_set

  def create_ansatz(self, random_list):
    circuit = QuantumCircuit(self.nqubit)
    for d in range(self.depth):
        circuit.add_gate(CNOT(0, 1))
        circuit.add_gate(merge(RY(0, random_list[self.gate_set*d+self.depth]), RZ(0, random_list[self.gate_set*d+(self.depth+1)])))
        circuit.add_gate(merge(RY(1, random_list[self.gate_set*d+(self.depth+2)]), RZ(1, random_list[self.gate_set*d+(self.depth+3)])))

    return circuit
