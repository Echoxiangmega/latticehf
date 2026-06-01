"""LatticeHF: Hartree-Fock solver for lattice models with intersite interactions."""

from .model import LatticeModel
from .hamiltonian import Hamiltonian, KSpaceHamiltonian
from .solver import HFSolver, KSpaceHFSolver
from .quantities import Quantities
from .green import (
    density_matrix_from_eigensystem,
    fermi_function,
    finite_temperature_density_matrix,
    green_density_residual,
    hartree_fock_self_energy,
    kspace_hartree_fock_self_energy,
    matsubara_green,
    zero_temperature_density_matrix,
)
from .order_parameters import (
    bond_current,
    bond_currents,
    bond_expectation,
    bond_order,
    charge_density,
    charge_order_amplitude,
    charge_structure_factor,
    loop_current,
    loop_current_pattern,
)
from .phase_diagram import (
    PeriodicHFResult,
    chemical_potential_for_filling,
    default_order_parameters,
    periodic_bond_current,
    periodic_loop_current,
    ruby_loop_current_order_parameters,
    scan_phase_diagram,
    solve_periodic_hf_finite_temperature,
    uniform_k_mesh,
)

__version__ = "0.1.0"

__all__ = [
    "LatticeModel",
    "Hamiltonian",
    "KSpaceHamiltonian",
    "HFSolver",
    "KSpaceHFSolver",
    "Quantities",
    "density_matrix_from_eigensystem",
    "fermi_function",
    "finite_temperature_density_matrix",
    "green_density_residual",
    "hartree_fock_self_energy",
    "kspace_hartree_fock_self_energy",
    "matsubara_green",
    "zero_temperature_density_matrix",
    "bond_current",
    "bond_currents",
    "bond_expectation",
    "bond_order",
    "charge_density",
    "charge_order_amplitude",
    "charge_structure_factor",
    "loop_current",
    "loop_current_pattern",
    "PeriodicHFResult",
    "chemical_potential_for_filling",
    "default_order_parameters",
    "periodic_bond_current",
    "periodic_loop_current",
    "ruby_loop_current_order_parameters",
    "scan_phase_diagram",
    "solve_periodic_hf_finite_temperature",
    "uniform_k_mesh",
]
