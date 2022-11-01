from typing import TYPE_CHECKING, Callable, cast

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


def ftol(ftol: float) -> Callable[[float, float], bool]:
    """Returns a function evaluating cost function tolerance.

    The return value is True when the cost function difference is less than ``ftol``;
    specifically ``|cost_prev - cost| <= 0.5 * ftol * (|cost_prev| + |cost|) + 1e-20``.
    """

    def fn(cost: float, cost_prev: float) -> bool:
        return (
            2.0 * abs(cost - cost_prev) <= ftol * (abs(cost) + abs(cost_prev)) + 1e-20
        )

    return fn


def gtol(gtol: float) -> Callable[["npt.NDArray[np.float_]"], bool]:
    """Returns a function evaluating gradient function tolerance.

    The return value is True when ``amax(abs(grad)) <= gtol``.
    """

    def fn(grad: "npt.NDArray[np.float_]") -> bool:
        return cast(float, np.amax(np.abs(grad))) <= gtol

    return fn
