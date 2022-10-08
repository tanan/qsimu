from abc import abstractproperty
from enum import Enum
from typing import Callable, List, Protocol, Union
from typing_extensions import TypeAlias
from qulacs import Observable

import numpy as np

from common.hamiltonian import Hamiltonian


Observables: TypeAlias = Union[Observable, List[Observable]]

class Estimate(Protocol):
    @abstractproperty
    def value(self) -> np.ndarray:
        ...

HamiltonianEstimator: TypeAlias = Callable[[Observables, Hamiltonian], Estimate]