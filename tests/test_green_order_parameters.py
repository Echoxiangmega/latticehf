import numpy as np

from latticehf import (
    Hamiltonian,
    LatticeModel,
    bond_current,
    charge_order_amplitude,
    hartree_fock_self_energy,
    loop_current,
    zero_temperature_density_matrix,
)


def test_hf_self_energy_matches_hamiltonian_fock_shift():
    model = LatticeModel(np.eye(2))
    model.add_orbital([0.0, 0.0])
    model.add_orbital([0.5, 0.0])
    model.add_hopping(0, 1, [0, 0], -1.0)
    model.add_v(0, 1, [0, 0], 0.7)

    hamiltonian = Hamiltonian(model, ncell=(1, 1))
    rho = np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex)
    fock = hamiltonian.compute_fock(rho)
    sigma = hartree_fock_self_energy(hamiltonian.nsite, hamiltonian.v_pairs, rho)

    assert np.allclose(fock, hamiltonian.H_hopping + sigma)


def test_zero_temperature_density_matrix_trace():
    h0 = np.diag([-2.0, -1.0, 3.0])
    rho, eigenvalues, eigenvectors = zero_temperature_density_matrix(h0, nelectron=2)

    assert np.allclose(np.trace(rho), 2.0)
    assert np.allclose(rho, np.diag([1.0, 1.0, 0.0]))
    assert eigenvalues.shape == (3,)
    assert eigenvectors.shape == (3, 3)


def test_charge_order_amplitude_for_staggered_density():
    rho = np.diag([0.75, 0.25, 0.75, 0.25])
    pattern = np.array([1.0, -1.0, 1.0, -1.0])

    assert np.isclose(charge_order_amplitude(rho, pattern), 0.25)


def test_loop_current_for_uniform_complex_triangle():
    hopping = np.array(
        [
            [0.0, -1.0, -1.0],
            [-1.0, 0.0, -1.0],
            [-1.0, -1.0, 0.0],
        ],
        dtype=complex,
    )
    rho = np.array(
        [
            [0.5, 0.0 - 0.1j, 0.0 + 0.1j],
            [0.0 + 0.1j, 0.5, 0.0 - 0.1j],
            [0.0 - 0.1j, 0.0 + 0.1j, 0.5],
        ],
        dtype=complex,
    )

    currents = [bond_current(rho, hopping, i, j) for i, j in [(0, 1), (1, 2), (2, 0)]]
    assert np.allclose(currents, currents[0])
    assert np.isclose(loop_current(rho, hopping, np.array([0, 1, 2])), currents[0])
