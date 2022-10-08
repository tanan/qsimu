from abc import abstractproperty
from enum import Enum
from typing import Callable, Protocol, Tuple, Union
from typing_extensions import TypeAlias
from collections.abc import Iterable

import numpy as np


class Hamiltonian(Protocol):
    @abstractproperty
    def value(self) -> np.ndarray:
        ...

    @abstractproperty
    def eigh(self) -> Tuple[np.ndarray, np.ndarray]:
        ...


class HamiltonianModel(Enum):
    TRANSVERSE_ISING = 1
    XY = 2
    HEISENBERG = 3


class HamiltonianModelError(Exception):
    ...


#: A type variable represents coefficients of each Hamiltonian term.
#: When you use Transverse Ising Hamiltonian(\sum a_j * X_j + \sum\sum J_jk * (Z_j Z_k) ), you can select Iterable[float, float].
#: First float is coefficient of X_j,  Second one is coefficient of Z_j Z_k.
Coefficients: TypeAlias = Union[
    Iterable[float], Tuple[Iterable[float], Iterable[float]]
]

#: TransverseIsingHamiltonianGenerator represents a function that generates Hamiltonian Vector
#: of a given params.
TransverseIsingHamiltonianGenerator: TypeAlias = Callable[[Coefficients], Hamiltonian]
