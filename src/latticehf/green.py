"""Green-function utilities for lattice Hartree-Fock calculations.

The existing HF solver updates the one-body density matrix by diagonalizing an
effective single-particle Hamiltonian.  This module exposes the same fixed point
in Green-function language:

    G(iw_n) = [(iw_n + mu) I - h0 - Sigma_HF[rho]]^{-1}
    rho     = <c^dagger_j c_i> = f(h0 + Sigma_HF)_{ij}

At zero temperature and fixed particle number, ``rho`` is the projector onto
occupied HF orbitals.  At finite temperature and fixed chemical potential, it is
obtained from the Fermi function.  These helpers are deliberately independent of
``HFSolver`` so they can also be reused by future GW/post-GW implementations.
"""

from __future__ import annotations

import numpy as np


def fermi_function(energies, beta=np.inf, mu=0.0):
    """Return Fermi occupations for single-particle energies."""
    energies = np.asarray(energies, dtype=float)
    if np.isinf(beta):
        occ = np.zeros_like(energies, dtype=float)
        occ[energies < mu] = 1.0
        occ[np.isclose(energies, mu)] = 0.5
        return occ

    x = beta * (energies - mu)
    occ = np.empty_like(x, dtype=float)
    occ[x > 40.0] = 0.0
    occ[x < -40.0] = 1.0
    mask = (x >= -40.0) & (x <= 40.0)
    occ[mask] = 1.0 / (np.exp(x[mask]) + 1.0)
    return occ


def density_matrix_from_eigensystem(eigenvalues, eigenvectors, nelectron=None, beta=np.inf, mu=0.0):
    """Construct rho_ij = <c_j^dagger c_i> from HF eigenvectors.

    Args:
        eigenvalues: one-body eigenvalues.
        eigenvectors: columns are eigenvectors in the site/orbital basis.
        nelectron: if given, fill the lowest ``nelectron`` orbitals at T=0.
        beta: inverse temperature for grand-canonical occupations.
        mu: chemical potential for grand-canonical occupations.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=complex)
    order = np.argsort(eigenvalues)
    vals = eigenvalues[order]
    vecs = eigenvectors[:, order]

    if nelectron is not None:
        if nelectron < 0 or nelectron > len(vals):
            raise ValueError("nelectron must lie between 0 and the Hilbert-space dimension")
        occ = np.zeros(len(vals), dtype=float)
        occ[:nelectron] = 1.0
    else:
        occ = fermi_function(vals, beta=beta, mu=mu)

    return (vecs * occ) @ vecs.conj().T


def zero_temperature_density_matrix(hamiltonian, nelectron):
    """Diagonalize a one-body Hamiltonian and return its T=0 density matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(hamiltonian, dtype=complex))
    rho = density_matrix_from_eigensystem(eigenvalues, eigenvectors, nelectron=nelectron)
    return rho, eigenvalues, eigenvectors


def finite_temperature_density_matrix(hamiltonian, beta, mu):
    """Return the grand-canonical density matrix f(H-mu) at finite temperature."""
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(hamiltonian, dtype=complex))
    rho = density_matrix_from_eigensystem(eigenvalues, eigenvectors, beta=beta, mu=mu)
    return rho, eigenvalues, eigenvectors


def matsubara_green(hamiltonian, omega_n, mu=0.0, self_energy=None):
    """Evaluate G(i omega_n) for a static or dynamic self-energy.

    ``self_energy`` may be ``None``, a matrix, or a callable accepting
    ``omega_n`` and returning a matrix.  This is the common interface needed by
    HF, GW, and post-GW-like one-shot calculations.
    """
    h = np.asarray(hamiltonian, dtype=complex)
    sigma = np.zeros_like(h)
    if self_energy is not None:
        sigma = self_energy(omega_n) if callable(self_energy) else np.asarray(self_energy, dtype=complex)
    z = 1j * omega_n + mu
    return np.linalg.inv(z * np.eye(h.shape[0], dtype=complex) - h - sigma)


def hartree_fock_self_energy(nsite, v_pairs, density_matrix):
    """Build the static HF self-energy from density-density interactions.

    ``v_pairs`` follows ``LatticeModel.get_v_pairs`` and contains triples
    ``(i, j, V_ij)``.  For spinless or spin-summed density matrices the result is

        Sigma_ii += V_ij rho_jj
        Sigma_jj += V_ij rho_ii
        Sigma_ij -= V_ij rho_ji
        Sigma_ji -= V_ij rho_ij

    which matches the convention used by ``Hamiltonian.compute_fock``.
    """
    rho = np.asarray(density_matrix, dtype=complex)
    sigma = np.zeros((nsite, nsite), dtype=complex)
    for i, j, v in v_pairs:
        sigma[i, i] += v * rho[j, j].real
        sigma[j, j] += v * rho[i, i].real
        sigma[i, j] -= v * rho[j, i]
        sigma[j, i] -= v * rho[i, j]
    return sigma


def kspace_hartree_fock_self_energy(model, k_points, density_matrix, ik):
    """Build the current k-space HF self-energy for one k point.

    This mirrors ``KSpaceHamiltonian.add_v_fock_k`` but exposes the result as
    ``Sigma_HF(k)`` so the periodic solver can be compared directly with the
    Green-function Dyson form ``G(k, iw)^{-1} = iw + mu - H0(k) - Sigma(k)``.

    The present k-space solver uses one unit-cell density matrix averaged over
    the sampled k points.  A future full translationally invariant Fock term can
    generalize this interface to a momentum convolution.
    """
    rho = np.asarray(density_matrix, dtype=complex)
    sigma = np.zeros((model.norb, model.norb), dtype=complex)
    rec_vecs = 2 * np.pi * np.linalg.inv(model.lat_vecs.T)
    k_cart = np.asarray(k_points[ik]) @ rec_vecs

    for term in model.v_terms:
        iorb = term["i"]
        jorb = term["j"]
        v = term["v"]
        delta = term["delta"]
        delta_R = delta @ model.lat_vecs
        phase = np.exp(1j * np.dot(k_cart, delta_R))
        v_eff = v * phase

        sigma[iorb, iorb] += v_eff * rho[jorb, jorb].real
        sigma[jorb, jorb] += v_eff * rho[iorb, iorb].real
        sigma[iorb, jorb] -= v_eff * rho[jorb, iorb]
        sigma[jorb, iorb] -= v_eff.conj() * rho[iorb, jorb]

    return sigma


def green_density_residual(h0, v_pairs, density_matrix, nelectron):
    """Return the HF fixed-point residual written as rho - f(h0+Sigma_HF)."""
    rho = np.asarray(density_matrix, dtype=complex)
    sigma = hartree_fock_self_energy(h0.shape[0], v_pairs, rho)
    rho_from_green, _, _ = zero_temperature_density_matrix(h0 + sigma, nelectron)
    return rho - rho_from_green
