from unittest import case
import numpy as np
from scipy.linalg import expm
import numpy.linalg as LA
import random
import sys
import datetime

from qiskit import QuantumCircuit
from qiskit import Aer, transpile,execute
from qiskit.quantum_info import random_pauli, state_fidelity, diamond_norm, Choi
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.tools.visualization import plot_histogram, plot_state_city

def create_hamiltonian(Nq, cn, gamma, Bn):
    '''
    create Hamiltonian gate
    target Hamiltonian is H=1/2*cn((1+gamma)XX+(1-gamma)YY)+BnZn
    Nq -> number of qubit
    cn -> coupling constant
    gamma -> parameter
    Bn -> Magnetic field
    '''
    
    XX= np.array(np.zeros(2**Nq))
    YY= np.array(np.zeros(2**Nq))
    Zn= np.array(np.zeros(2**Nq))
    Identity = 'I' * Nq
    hamiltonian = np.array(np.zeros(2**Nq))
    
    for k in range(0, Nq):
        '''
        隣あうqubitのみ相互作用するようなモデルを考える
        0 1 2 ... k-1 X X k+2 ... N-1
        0 1 2 ... k-1 Y Y k+2 ... N-1
        '''
        
        # 端まで計算したらスキップ
        if k == Nq-1:
            continue

        hamiX = Pauli(Identity[:k] + 'XX' + Identity[k+2:])
        hamiY = Pauli(Identity[:k] + 'YY' + Identity[k+2:])

        XX = XX + 0.5*cn[k]*(1+gamma)*hamiX.to_matrix()
        YY = YY + 0.5*cn[k]*(1-gamma)*hamiY.to_matrix()

    for k in range(0, Nq):
        hamiZ = Pauli(Identity[:k] + 'Z' + Identity[k+1:])
        Zn = Zn + Bn[k] * hamiZ.to_matrix()
    
    return XX + YY + Zn


def create_choi(t_list, q, cn, r, bn, direct=True):
    '''
    loop -> loop count
    q -> number of qubit
    cn -> coupling constant
    r -> gamma parameter
    bn -> magnetic field
    direct -> direct or indirect control
    '''
    
    qc = QuantumCircuit(q)
    random.shuffle(t_list)
    for t in t_list:
        # Hamiltonianの時間発展を計算
        hami = expm(-1j*create_hamiltonian(q,cn,r,bn)*t)
        qc.append(Operator(hami),list(range(q)))

        # 各bitに独立な確率でPauliゲートを追加
        if direct:
            qc.append(random_pauli(1), [random.randint(0,q-1)])
        else:
            qc.append(random_pauli(1), [random.randint(0,1)])

    return Choi(qc).data


def run(loop, t_direct, t_indirect):
    qubit = 4
    cn = [1] * qubit #[1,1,1,1,1]
    r = 0
    bn = [0] * qubit #[0,0,0,0,0]

    choi_qubit = 2 * qubit
    choi_direct = np.array(np.zeros(2**choi_qubit))
    choi_indirect = np.array(np.zeros(2**choi_qubit))

    for i in range(0, loop):
        print("%s: " % datetime.datetime.now())
        choi_direct = choi_direct + create_choi(t_direct, qubit, cn, r, bn, True)
        choi_indirect = choi_indirect + create_choi(t_indirect, qubit, cn, r, bn, False)

    eps = (choi_direct - choi_indirect) / loop
    print(diamond_norm(Choi(eps)))

if __name__ == '__main__':
  args = sys.argv
  gate_num = args[1]
  loop = args[2]
  print("loop: %s" % loop)
  if gate_num == '20':
    print("gate_num: %s" % gate_num)
    t_direct = [0.005, 0.005, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.99, 1, 2, 2]
  elif gate_num == '40':
    print("gate_num: %s" % gate_num)
    t_direct = [0.005, 0.005, 0.005, 0.005, 0.025, 0.025, 0.025, 0.025, 0.05, 0.05, 0.05, 0.075, 0.075, 0.075, 0.075, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 1, 1, 1, 1]
  elif gate_num == '60':
    print("gate_num: %s" % gate_num)
    t_direct = [0.005, 0.005, 0.005, 0.005, 0.005, 0.015, 0.015, 0.015, 0.015, 0.015, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5]
  else:
    sys.exit("gate_num is not set.")

  t_indirect = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.3, 0.4, 0.4, 0.4]
  run(int(loop), t_direct, t_indirect)