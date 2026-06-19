"""Sous-package causal : NOTEARS, RKHS, do-calculus.

L4 : découverte causale avec DAG garanti acyclique (NOTEARS), opérateur RKHS
via Random Fourier Features, et vrai do-calculus de Pearl.
"""

from .notears import notears_penalty
from .rkhs import RKHSCausalOperator
from .do import do_intervention

__all__ = ["notears_penalty", "RKHSCausalOperator", "do_intervention"]
