"""Order-parameter diagnostics for periodic lattice mean-field states.

The functions operate on the one-body density matrix

    rho_ij = <c_j^dagger c_i>

which is the natural output of both HF and Green-function calculations.  This
keeps charge order, bond order, and loop-current diagnostics independent of the
solver used to obtain ``rho``.
"""

from __future__ import annotations

import numpy as np


def charge_density(density_matrix):
    """Return n_i = rho_ii."""
    return np.real(np.diag(np.asarray(density_matrix, dtype=complex)))


def charge_order_amplitude(density_matrix, pattern=None):
    """Measure charge modulation amplitude.

    If ``pattern`` is None, return the root-mean-square density modulation
    around the spatial average.  If ``pattern`` is supplied, return the density
    projected onto that real-space pattern, normalized by ``sum |pattern|^2``.
    """
    density = charge_density(density_matrix)
    centered = density - np.mean(density)
    if pattern is None:
        return float(np.sqrt(np.mean(centered**2)))

    pattern = np.asarray(pattern, dtype=float)
    if pattern.shape != density.shape:
        raise ValueError("pattern must have the same length as the site density")
    norm = np.vdot(pattern, pattern).real
    if norm == 0.0:
        raise ValueError("pattern must not be identically zero")
    return float(np.vdot(pattern, centered).real / norm)


def charge_structure_factor(density_matrix, positions, q_points, connected=True):
    """Return |sum_i exp(i q.r_i) n_i|^2 / N for each q.

    With ``connected=True`` the uniform average density is subtracted first,
    which makes charge-order peaks easier to see.
    """
    density = charge_density(density_matrix)
    if connected:
        density = density - np.mean(density)
    positions = np.asarray(positions, dtype=float)
    q_points = np.atleast_2d(np.asarray(q_points, dtype=float))
    values = []
    for q in q_points:
        rho_q = np.sum(np.exp(1j * (positions @ q)) * density)
        values.append(np.abs(rho_q) ** 2 / len(density))
    return np.asarray(values, dtype=float)


def bond_expectation(density_matrix, i, j):
    """Return <c_i^dagger c_j> for an oriented bond i <- j."""
    rho = np.asarray(density_matrix, dtype=complex)
    return rho[j, i]


def bond_order(density_matrix, bonds):
    """Return real bond orders Re <c_i^dagger c_j> for ``[(i, j), ...]``."""
    return np.asarray([bond_expectation(density_matrix, i, j).real for i, j in bonds], dtype=float)


def bond_current(density_matrix, hopping_matrix, i, j, convention="continuity"):
    """Return the oriented current on bond i -> j.

    For a one-body Hamiltonian H0 = sum_ij h_ij c_i^dagger c_j, the continuity
    equation gives J_{i->j} = 2 Im[h_ij <c_i^dagger c_j>] in units e = hbar = 1.
    Some papers use the opposite sign; set ``convention='opposite'`` to flip it.
    """
    rho = np.asarray(density_matrix, dtype=complex)
    h = np.asarray(hopping_matrix, dtype=complex)
    value = 2.0 * np.imag(h[i, j] * rho[j, i])
    if convention == "opposite":
        value = -value
    elif convention != "continuity":
        raise ValueError("convention must be 'continuity' or 'opposite'")
    return float(value)


def bond_currents(density_matrix, hopping_matrix, bonds, convention="continuity"):
    """Return oriented bond currents for ``[(i, j), ...]``."""
    return np.asarray(
        [bond_current(density_matrix, hopping_matrix, i, j, convention=convention) for i, j in bonds],
        dtype=float,
    )


def loop_current(density_matrix, hopping_matrix, loop, convention="continuity"):
    """Return the average circulating current around an oriented loop.

    ``loop`` is an ordered list of sites, for example ``[0, 1, 2]`` for a
    triangle.  The closing bond from the last site back to the first is included.
    A nonzero value diagnoses time-reversal-breaking loop-current order when
    the loop orientation is chosen consistently across the unit cell.
    """
    loop = list(loop)
    if len(loop) < 3:
        raise ValueError("loop must contain at least three sites")
    bonds = list(zip(loop, loop[1:] + loop[:1]))
    return float(np.mean(bond_currents(density_matrix, hopping_matrix, bonds, convention=convention)))


def loop_current_pattern(density_matrix, hopping_matrix, loops, signs=None, convention="continuity"):
    """Project loop currents onto a candidate pattern.

    ``loops`` is a list of oriented loops.  ``signs`` can encode ferro-current,
    staggered-current, or Ruby-lattice-specific patterns.  If omitted, the
    function returns all loop currents without projection.
    """
    currents = np.asarray(
        [loop_current(density_matrix, hopping_matrix, loop, convention=convention) for loop in loops],
        dtype=float,
    )
    if signs is None:
        return currents
    signs = np.asarray(signs, dtype=float)
    if signs.shape != currents.shape:
        raise ValueError("signs must have the same length as loops")
    norm = np.vdot(signs, signs).real
    if norm == 0.0:
        raise ValueError("signs must not be identically zero")
    return float(np.vdot(signs, currents).real / norm)
