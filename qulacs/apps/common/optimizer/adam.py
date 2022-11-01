from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional

from common.optimizer import OptimizerState, OptimizerStatus
from .tolerance import ftol as create_ftol

_const_zero_array = np.zeros(0, dtype=float)


@dataclass(frozen=True)
class OptimizerStateAdam(OptimizerState):
    m: np.ndarray = _const_zero_array
    v: np.ndarray = _const_zero_array


class Adam:
    def __init__(self, lr=0.1, betas=(0.9, 0.999), eps=1e-6, ftol=1e-5) -> None:
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self._ftol: Optional[Callable[[float, float], bool]] = None
        if ftol is not None:
            if not 0.0 < ftol:
                raise ValueError("ftol must be a positive float.")
            self._ftol = create_ftol(ftol)

    def get_init_state(self, init_params) -> OptimizerStateAdam:
        params = np.array(init_params)
        zeros = np.zeros(len(params), dtype=float)
        return OptimizerStateAdam(
            params=params,
            m=zeros,
            v=zeros,
        )

    def step(
        self,
        state: OptimizerState,
        cost_fn,
        grad_fn,
    ):
        funcalls = state.funcalls
        gradcalls = state.gradcalls
        niter = state.niter + 1

        if niter == 1:
            cost_prev = cost_fn(state.params)
            funcalls += 1
        else:
            cost_prev = state.cost

        beta1, beta2 = self.betas

        grads = grad_fn(state.params)
        gradcalls += 1

        lr_t = self.lr * np.sqrt(1.0 - beta2**niter) / (1.0 - beta1**niter)
        m = beta1 * state.m + (1 - beta1) * grads
        v = beta2 * state.v + (1 - beta2) * (grads**2)
        params = state.params - lr_t * m / (np.sqrt(v) + 1e-7)

        cost = cost_fn(params)
        funcalls += 1

        if self._ftol and self._ftol(cost, cost_prev):
            status = OptimizerStatus.CONVERGED
        else:
            status = OptimizerStatus.SUCCESS

        return OptimizerStateAdam(
            params=state.params,
            cost=cost,
            status=status,
            niter=niter,
            funcalls=funcalls,
            gradcalls=gradcalls,
            m=m,
            v=v,
        )
