"""Finite-temperature periodic HF scans and phase diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .green import kspace_hartree_fock_self_energy
from .hamiltonian import KSpaceHamiltonian
from .order_parameters import charge_order_amplitude


def uniform_k_mesh(nk1: int, nk2: int, shift=(0.0, 0.0)) -> np.ndarray:
    """Return a uniform reduced-coordinate mesh in the Brillouin zone."""
    if nk1 <= 0 or nk2 <= 0:
        raise ValueError("nk1 and nk2 must be positive")
    points = []
    for i in range(nk1):
        for j in range(nk2):
            points.append([(i + shift[0]) / nk1, (j + shift[1]) / nk2])
    return np.asarray(points, dtype=float)


@dataclass
class PeriodicHFResult:
    """Result of a translationally invariant finite-temperature HF solve."""

    converged: bool
    iterations: int
    density_matrix: np.ndarray
    density_k: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    chemical_potential: float
    filling: float
    target_particles: float
    mean_field_energy: float
    history: list[dict]


def _target_particles(filling: float, norb: int, filling_unit: str) -> float:
    if filling_unit == "fraction":
        return float(filling) * norb
    if filling_unit == "particles":
        return float(filling)
    raise ValueError("filling_unit must be 'fraction' or 'particles'")


def _fermi(x):
    out = np.empty_like(x, dtype=float)
    out[x > 40.0] = 0.0
    out[x < -40.0] = 1.0
    mask = (x >= -40.0) & (x <= 40.0)
    out[mask] = 1.0 / (np.exp(x[mask]) + 1.0)
    return out


def chemical_potential_for_filling(eigenvalues, beta: float, target_particles: float) -> tuple[float, np.ndarray]:
    """Find mu so average_k sum_band f(beta*(epsilon-mu)) hits target filling."""
    values = np.asarray(eigenvalues, dtype=float)
    if target_particles < 0.0 or target_particles > values.shape[1]:
        raise ValueError("target filling must lie between 0 and the number of orbitals")

    if target_particles == 0.0:
        mu = float(np.min(values) - 100.0 / max(beta, 1e-12))
        return mu, np.zeros_like(values)
    if target_particles == values.shape[1]:
        mu = float(np.max(values) + 100.0 / max(beta, 1e-12))
        return mu, np.ones_like(values)

    width = 100.0 / max(beta, 1e-12) + 10.0
    lo = float(np.min(values) - width)
    hi = float(np.max(values) + width)
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        occ = _fermi(beta * (values - mu))
        particles = np.sum(occ) / values.shape[0]
        if abs(particles - target_particles) < 1e-12:
            return float(mu), occ
        if particles < target_particles:
            lo = mu
        else:
            hi = mu
    mu = 0.5 * (lo + hi)
    occ = _fermi(beta * (values - mu))
    return mu, occ


def density_from_periodic_eigensystem(eigenvalues, eigenvectors, beta: float, target_particles: float):
    """Return rho(k), average rho, chemical potential, and occupations."""
    mu, occupations = chemical_potential_for_filling(eigenvalues, beta, target_particles)
    nk, norb = eigenvalues.shape
    density_k = np.zeros((nk, norb, norb), dtype=complex)
    for ik in range(nk):
        density_k[ik] = (eigenvectors[ik] * occupations[ik]) @ eigenvectors[ik].conj().T
    return density_k, np.mean(density_k, axis=0), mu, occupations


def solve_periodic_hf_finite_temperature(
    model,
    k_points,
    filling: float,
    beta: float,
    filling_unit: str = "fraction",
    initial_density=None,
    max_iter: int = 200,
    tol: float = 1e-8,
    mixing: float = 0.5,
):
    """Solve translationally invariant HF at fixed temperature and filling.

    The current approximation follows the repository's k-space HF convention:
    the interaction self-energy is built from a unit-cell density matrix averaged
    over the sampled Brillouin zone.  The output keeps both rho(k) and the
    averaged rho so future momentum-convolution Fock terms can reuse the API.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if not (0.0 < mixing <= 1.0):
        raise ValueError("mixing must lie in (0, 1]")

    k_points = np.asarray(k_points, dtype=float)
    hamiltonian_k = KSpaceHamiltonian(model, k_points)
    nk = len(k_points)
    norb = model.norb
    target = _target_particles(filling, norb, filling_unit)

    if initial_density is None:
        density = np.eye(norb, dtype=complex) * (target / norb)
    else:
        density = np.asarray(initial_density, dtype=complex)
        if density.shape != (norb, norb):
            raise ValueError("initial_density must have shape (norb, norb)")

    history = []
    converged = False
    last = None

    for iteration in range(1, max_iter + 1):
        fock_k = np.zeros((nk, norb, norb), dtype=complex)
        eigenvalues = np.zeros((nk, norb), dtype=float)
        eigenvectors = np.zeros((nk, norb, norb), dtype=complex)

        for ik in range(nk):
            sigma = kspace_hartree_fock_self_energy(model, k_points, density, ik)
            fock_k[ik] = hamiltonian_k.get_hopping(ik) + sigma
            vals, vecs = np.linalg.eigh(fock_k[ik])
            eigenvalues[ik] = vals
            eigenvectors[ik] = vecs

        density_k, new_density, mu, occupations = density_from_periodic_eigensystem(
            eigenvalues,
            eigenvectors,
            beta,
            target,
        )
        mixed_density = mixing * new_density + (1.0 - mixing) * density
        diff = float(np.max(np.abs(mixed_density - density)))
        density = 0.5 * (mixed_density + mixed_density.conj().T)
        kinetic_energy = _periodic_kinetic_energy(hamiltonian_k, density_k)
        interaction_energy = _periodic_interaction_energy(model, k_points, density, density_k)
        mean_field_energy = kinetic_energy + interaction_energy
        last = (density_k, eigenvalues, eigenvectors, mu, mean_field_energy)
        history.append({"iteration": iteration, "diff": diff, "mu": mu, "energy": mean_field_energy})

        if diff < tol:
            converged = True
            break

    density_k, eigenvalues, eigenvectors, mu, mean_field_energy = last
    return PeriodicHFResult(
        converged=converged,
        iterations=iteration,
        density_matrix=density,
        density_k=density_k,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        chemical_potential=float(mu),
        filling=float(filling),
        target_particles=float(target),
        mean_field_energy=float(mean_field_energy),
        history=history,
    )


def _periodic_kinetic_energy(hamiltonian_k: KSpaceHamiltonian, density_k) -> float:
    energy = 0.0
    for ik in range(hamiltonian_k.nk):
        energy += np.trace(density_k[ik] @ hamiltonian_k.get_hopping(ik)).real
    return float(energy / hamiltonian_k.nk)


def _periodic_interaction_energy(model, k_points, density, density_k) -> float:
    energy = 0.0
    for term in model.v_terms:
        i = term["i"]
        j = term["j"]
        v = term["v"]
        energy += 0.5 * v * density[i, i].real * density[j, j].real
        energy -= 0.5 * v * abs(_periodic_bond_expectation(model, k_points, density_k, i, j, term["delta"])) ** 2
    return float(energy)


def _find_hopping(model, i, j, delta):
    delta = np.asarray(delta, dtype=int)
    for term in model.hopping_terms:
        if term["i"] == i and term["j"] == j and np.array_equal(term["delta"], delta):
            return term["t"]
    for term in model.hopping_terms:
        if term["i"] == j and term["j"] == i and np.array_equal(term["delta"], -delta):
            return np.conj(term["t"])
    raise ValueError(f"No hopping found for oriented bond ({i}, {j}, delta={delta.tolist()})")


def _bond_phase(model, k_frac, delta):
    rec_vecs = 2 * np.pi * np.linalg.inv(model.lat_vecs.T)
    k_cart = np.asarray(k_frac) @ rec_vecs
    delta_R = np.asarray(delta) @ model.lat_vecs
    return np.exp(1j * np.dot(k_cart, delta_R))


def _periodic_bond_expectation(model, k_points, density_k, i, j, delta):
    value = 0.0j
    for ik, k_frac in enumerate(k_points):
        value += _bond_phase(model, k_frac, delta) * density_k[ik, j, i]
    return value / len(k_points)


def periodic_bond_current(model, k_points, density_k, i, j, delta):
    """Return the oriented current on a periodic bond i -> j + delta."""
    hopping = _find_hopping(model, i, j, delta)
    expectation = _periodic_bond_expectation(model, k_points, density_k, i, j, delta)
    return float(2.0 * np.imag(hopping * expectation))


def periodic_loop_current(model, k_points, density_k, loop):
    """Average oriented current around a loop of ``(i, j, delta)`` bonds."""
    currents = [periodic_bond_current(model, k_points, density_k, i, j, delta) for i, j, delta in loop]
    return float(np.mean(currents))


def ruby_loop_current_order_parameters(model, k_points, density_k):
    """Return Ruby-lattice same/opposite triangle loop-current amplitudes."""
    upper = [(0, 1, [0, 0]), (1, 2, [0, 0]), (2, 0, [0, 0])]
    lower = [(3, 4, [0, 0]), (4, 5, [0, 0]), (5, 3, [0, 0])]
    upper_current = periodic_loop_current(model, k_points, density_k, upper)
    lower_current = periodic_loop_current(model, k_points, density_k, lower)
    return {
        "loop_upper": upper_current,
        "loop_lower": lower_current,
        "loop_same": 0.5 * (upper_current + lower_current),
        "loop_opposite": 0.5 * (upper_current - lower_current),
    }


def default_order_parameters(result: PeriodicHFResult, model, k_points):
    orders = {
        "charge_order": charge_order_amplitude(result.density_matrix),
    }
    if model.norb >= 6:
        orders.update(ruby_loop_current_order_parameters(model, k_points, result.density_k))
    return orders


def classify_orders(order_parameters, threshold: float = 1e-5):
    active = []
    if abs(order_parameters.get("charge_order", 0.0)) > threshold:
        active.append("charge")
    if abs(order_parameters.get("loop_same", 0.0)) > threshold:
        active.append("loop_same")
    if abs(order_parameters.get("loop_opposite", 0.0)) > threshold:
        active.append("loop_opposite")
    return "+".join(active) if active else "normal"


def scan_phase_diagram(
    model_factory: Callable[[float], object],
    k_points,
    interaction_values,
    fillings,
    beta: float,
    filling_unit: str = "fraction",
    order_parameter_fn=default_order_parameters,
    threshold: float = 1e-5,
    max_iter: int = 200,
    tol: float = 1e-8,
    mixing: float = 0.5,
    warm_start: bool = True,
):
    """Scan a V-filling grid and return periodic finite-temperature HF phases."""
    k_points = np.asarray(k_points, dtype=float)
    interaction_values = np.asarray(interaction_values, dtype=float)
    fillings = np.asarray(fillings, dtype=float)
    results = {}
    phase_labels = np.empty((len(interaction_values), len(fillings)), dtype=object)
    charge_order = np.zeros((len(interaction_values), len(fillings)), dtype=float)
    loop_same = np.zeros_like(charge_order)
    loop_opposite = np.zeros_like(charge_order)

    previous_by_filling = [None for _ in fillings]
    for i_v, interaction in enumerate(interaction_values):
        for i_n, filling in enumerate(fillings):
            model = model_factory(float(interaction))
            initial = previous_by_filling[i_n] if warm_start else None
            result = solve_periodic_hf_finite_temperature(
                model,
                k_points,
                filling=float(filling),
                beta=beta,
                filling_unit=filling_unit,
                initial_density=initial,
                max_iter=max_iter,
                tol=tol,
                mixing=mixing,
            )
            orders = order_parameter_fn(result, model, k_points)
            label = classify_orders(orders, threshold=threshold)
            results[(float(interaction), float(filling))] = {
                "result": result,
                "orders": orders,
                "phase": label,
            }
            phase_labels[i_v, i_n] = label
            charge_order[i_v, i_n] = orders.get("charge_order", 0.0)
            loop_same[i_v, i_n] = orders.get("loop_same", 0.0)
            loop_opposite[i_v, i_n] = orders.get("loop_opposite", 0.0)
            previous_by_filling[i_n] = result.density_matrix.copy()

    return {
        "interaction_values": interaction_values,
        "fillings": fillings,
        "phase_labels": phase_labels,
        "charge_order": charge_order,
        "loop_same": loop_same,
        "loop_opposite": loop_opposite,
        "points": results,
    }
