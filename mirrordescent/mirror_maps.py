import abc
import typing as t

import numpy as np
import numpy.typing as npt

class MirrorMap(abc.ABC):
    """
    Abstract base class for mirror maps.
    """


class EntropicMirrorMap(MirrorMap):
    """
    The entropic mirror map; see eqn. (3.4) in "Mirrored Langevin Dynamics".
    """
    def __call__(self, x: npt.ArrayLike) -> np.float64:
        def _sum_x_log_x(arr: npt.ArrayLike) -> np.float64:
            # Sum xlogx with the convention that 0log0 = 0 
            return np.sum(np.where(arr > 0, arr * np.log(arr), 0))
        
        first_term = _sum_x_log_x(x)
        second_term = _sum_x_log_x(np.array([1-sum(x)]))
        return first_term + second_term
    
    def fenchel_dual(self, y: npt.ArrayLike) -> np.float64:
        sum_exp_y = np.sum(np.exp(y))
        return np.log(1 + sum_exp_y)