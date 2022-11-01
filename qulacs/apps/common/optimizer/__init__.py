from enum import Enum, auto

import numpy as np
from dataclasses import dataclass


class OptimizerStatus(Enum):
    """Status of optimization."""

    #: No error, not converged yet.
    SUCCESS = auto()
    #: The optimization failed and cannot be continued.
    FAILED = auto()
    #: The optimization converged.
    CONVERGED = auto()


@dataclass(frozen=True)
class OptimizerState:
    params: np.ndarray
    cost: float = 0.0
    status: OptimizerStatus = OptimizerStatus.SUCCESS
    niter: int = 0
    funcalls: int = 0
    gradcalls: int = 0

    @property
    def n_params(self) -> int:
        return len(self.params)
