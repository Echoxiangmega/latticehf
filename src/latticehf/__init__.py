"""LatticeHF: Hartree-Fock solver for lattice models with intersite interactions."""

from .model import LatticeModel
from .hamiltonian import Hamiltonian, KSpaceHamiltonian
from .solver import HFSolver, KSpaceHFSolver
from .quantities import Quantities

__version__ = "0.1.0"

__all__ = [
    "LatticeModel",
    "Hamiltonian",
    "KSpaceHamiltonian",
    "HFSolver",
    "KSpaceHFSolver",
    "Quantities",
]