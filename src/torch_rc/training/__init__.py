"""ESN training utilities.

This module provides trainers for ESN models that fit readout layers
algebraically (not via SGD).
"""

from .trainer import ESNTrainer

__all__ = ["ESNTrainer"]
