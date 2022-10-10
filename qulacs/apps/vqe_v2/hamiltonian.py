from openfermion.ops import QubitOperator
from qulacs.observable import create_observable_from_openfermion_text


def create_ising_hamiltonian(nqubit):
    return create_observable_from_openfermion_text(str(create_qubit_operator(nqubit)))


def create_qubit_operator(nqubit):
    hami = QubitOperator()
    for i in range(nqubit):
        hami = hami + (-1.0 * QubitOperator("X" + str(i)))
        for j in range(i + 1, nqubit):
            if i + 1 == j:
                hami = hami + (-1.0 * QubitOperator("Z" + str(i) + " " + "Z" + str(j)))

    return hami
