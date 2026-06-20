"""Sous-package reasoning : preuves, conjectures, ACT, self-consistency.

L5 : raisonnement mathematical with verification exact (soundness garantie).
"""

from .proof import (
    InferenceRule, ProofStep, Proof,
    ProofGenerator, ProofVerifier, ProofReward,
    all_rules,
)
from .conjecture import (
    ConjectureTemplate, Conjecture,
    ConjectureGenerator, ConjectureTester, ConjectureMemory, ConjectureDiscoveryLoop,
)
from .act import RecursiveReasoner
from .self_consistency import SelfConsistencyCheck

__all__ = [
    "InferenceRule", "ProofStep", "Proof",
    "ProofGenerator", "ProofVerifier", "ProofReward", "all_rules",
    "ConjectureTemplate", "Conjecture",
    "ConjectureGenerator", "ConjectureTester", "ConjectureMemory", "ConjectureDiscoveryLoop",
    "RecursiveReasoner", "SelfConsistencyCheck",
]
