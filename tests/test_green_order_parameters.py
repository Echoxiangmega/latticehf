import numpy as np

from latticehf import (
    Hamiltonian,
    HFSolver,
    KSpaceHamiltonian,
    LatticeModel,
    bond_current,
    charge_order_amplitude,
    hartree_fock_self_energy,
    kspace_hartree_fock_self_energy,
    loop_current,
    zero_temperature_density_matrix,
)


def _add_hermitian_hopping(model, i, j, delta, amplitude):
    delta = np.asarray(delta, dtype=int)
    model.add_hopping(i, j, delta, amplitude)
    model.add_hopping(j, i, -delta, np.conj(amplitude))


def _ruby_lattice_model(add_conjugate_hoppings=True):
    lat = np.array([[np.sqrt(3) / 2, 1 / 2], [-np.sqrt(3) / 2, 1 / 2]])
    model = LatticeModel(lat)

    orbitals = [
        [1 / 3 - 1 / 8, 2 / 3 + 1 / 8],
        [1 / 3 + 2 / 8, 2 / 3 + 1 / 8],
        [1 / 3 - 1 / 8, 2 / 3 - 2 / 8],
        [2 / 3 + 1 / 8, 1 / 3 - 1 / 8],
        [2 / 3 + 1 / 8, 1 / 3 + 2 / 8],
        [2 / 3 - 2 / 8, 1 / 3 - 1 / 8],
    ]
    for position in orbitals:
        model.add_orbital(position)

    ti, t1 = 0.4, 0.2
    for i, j, delta, amplitude in [
        (0, 1, [0, 0], ti),
        (0, 2, [0, 0], ti),
        (2, 1, [0, 0], ti),
        (3, 4, [0, 0], ti),
        (3, 5, [0, 0], ti),
        (4, 5, [0, 0], ti),
        (1, 4, [0, 0], t1),
        (5, 0, [0, -1], t1),
        (2, 3, [-1, 0], t1),
        (3, 1, [0, -1], t1),
        (2, 5, [0, 0], t1),
        (0, 4, [-1, 0], t1),
    ]:
        if add_conjugate_hoppings:
            _add_hermitian_hopping(model, i, j, delta, amplitude)
        else:
            model.add_hopping(i, j, delta, amplitude)

    model.add_v(0, 1, [0, 0], 0.3)
    model.add_v(1, 4, [0, 0], 0.2)
    model.add_v(0, 4, [-1, 0], 0.15)
    return model


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


def test_ruby_lattice_hf_matches_green_function_fixed_point_step():
    model = _ruby_lattice_model()
    hamiltonian = Hamiltonian(model, ncell=(2, 2))
    nelectron = 8

    initial_rho = np.eye(hamiltonian.nsite, dtype=complex) * (nelectron / hamiltonian.nsite)
    fock_from_original = hamiltonian.compute_fock(initial_rho)
    sigma_from_green = hartree_fock_self_energy(hamiltonian.nsite, hamiltonian.v_pairs, initial_rho)

    assert np.allclose(fock_from_original, hamiltonian.H_hopping + sigma_from_green)

    rho_from_green, _, _ = zero_temperature_density_matrix(
        hamiltonian.H_hopping + sigma_from_green,
        nelectron=nelectron,
    )
    solver = HFSolver(hamiltonian, nelectron=nelectron)
    solver.density_matrix = initial_rho.copy()
    solver.solve(max_iter=1, verbose=False)

    assert np.allclose(solver.density_matrix, rho_from_green)
    assert np.isclose(
        hamiltonian.compute_energy(solver.density_matrix),
        hamiltonian.compute_energy(rho_from_green),
    )


def test_periodic_ruby_lattice_kspace_hf_matches_green_function_self_energy():
    model = _ruby_lattice_model(add_conjugate_hoppings=False)
    k_points = np.array(
        [
            [0.0, 0.0],
            [1 / 3, 1 / 3],
            [1 / 2, 0.0],
            [1 / 4, 1 / 4],
        ]
    )
    hamiltonian_k = KSpaceHamiltonian(model, k_points)
    density_k = np.eye(model.norb, dtype=complex) * 0.5

    for ik in range(len(k_points)):
        sigma_green = kspace_hartree_fock_self_energy(model, k_points, density_k, ik)
        sigma_original = np.zeros((model.norb, model.norb), dtype=complex)
        for v_term in model.v_terms:
            sigma_original += hamiltonian_k.add_v_fock_k(density_k, v_term, ik)

        fock_original = hamiltonian_k.get_hopping(ik) + sigma_original
        fock_green = hamiltonian_k.get_hopping(ik) + sigma_green

        assert np.allclose(sigma_original, sigma_green)
        assert np.allclose(fock_original, fock_green)


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
