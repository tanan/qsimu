## 基本ゲート
import numpy as np
from qulacs.gate import X, Z, DenseMatrix
from functools import reduce

I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()

## fullsizeのgateをつくる関数.
def make_fullgate(list_SiteAndOperator, nqubit):
    '''
    list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...] を受け取り,
    関係ないqubitにIdentityを挿入して
    I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    という(2**nqubit, 2**nqubit)行列をつくる.
    '''
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = [] ## 1-qubit gateを並べてnp.kronでreduceする
    cnt = 0
    for i in range(nqubit):
        if (i in list_Site):
            list_SingleGates.append( list_SiteAndOperator[cnt][1] )
            cnt += 1
        else: ## 何もないsiteはidentity
            list_SingleGates.append(I_mat)

    return reduce(np.kron, list_SingleGates)

def create_hamiltonian_gate(nqubit, time_step):
  ham = np.zeros((2**nqubit,2**nqubit), dtype = complex)
  for i in range(nqubit): ## i runs 0 to nqubit-1
      Jx = -1. + 2.*np.random.rand() ## -1~1の乱数
      ham += Jx * make_fullgate( [ [i, X_mat] ], nqubit)
      for j in range(i+1, nqubit):
          J_ij = -1. + 2.*np.random.rand()
          ham += J_ij * make_fullgate ([ [i, Z_mat], [j, Z_mat]], nqubit)

  ## 対角化して時間発展演算子をつくる. H*P = P*D <-> H = P*D*P^dagger
  diag, eigen_vecs = np.linalg.eigh(ham)
  time_evol_op = np.dot(np.dot(eigen_vecs, np.diag(np.exp(-1j*time_step*diag))), eigen_vecs.T.conj()) # e^-iHT
  return DenseMatrix([i for i in range(nqubit)], time_evol_op)