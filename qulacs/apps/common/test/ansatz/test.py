import pytest
from common.ansatz import XYAnsatz, AnsatzType

class TestXYAnsatz:
    def test_ansatz_type(self) -> None:
        time = {"type": "random"}
        bn = {"value": [0,0,0,0]}
        ansatz = XYAnsatz(nqubit=4, depth=1, gate_set=4, time=time, bn=bn, noise=None)
        typ = ansatz.ansatz_type()
        assert typ == AnsatzType.INDIRECT_XY

    def test_create_hamiltonian(self) -> None:
        time = {"type": "random"}
        bn = {"value": [0,0,0]}
        cn = [1.0] * 3
        gamma = 0
        ansatz = XYAnsatz(nqubit=3, depth=1, gate_set=4, time=time, bn=bn, noise=None)
        diag, eigen_vecs = ansatz.create_hamiltonian(cn, bn, gamma)
        print(eigen_vecs)
        assert diag[0] == pytest.approx(-1.41421356)
