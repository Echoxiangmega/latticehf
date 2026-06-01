import numpy as np

from latticehf import (
    chemical_potential_for_filling,
    scan_phase_diagram,
    solve_periodic_hf_finite_temperature,
    uniform_k_mesh,
)
from test_green_order_parameters import _ruby_lattice_model


def test_chemical_potential_hits_target_filling():
    eigenvalues = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]])
    mu, occupations = chemical_potential_for_filling(eigenvalues, beta=20.0, target_particles=1.5)

    assert np.isfinite(mu)
    assert np.isclose(np.sum(occupations) / eigenvalues.shape[0], 1.5, atol=1e-10)


def test_periodic_hf_finite_temperature_ruby_returns_target_filling():
    model = _ruby_lattice_model(add_conjugate_hoppings=False)
    k_points = uniform_k_mesh(2, 2)

    result = solve_periodic_hf_finite_temperature(
        model,
        k_points,
        filling=0.5,
        beta=10.0,
        max_iter=20,
        tol=1e-7,
    )

    assert result.density_matrix.shape == (model.norb, model.norb)
    assert result.density_k.shape == (len(k_points), model.norb, model.norb)
    assert np.isclose(np.trace(result.density_matrix).real, 0.5 * model.norb, atol=1e-8)
    assert np.isfinite(result.chemical_potential)


def test_scan_phase_diagram_ruby_small_grid():
    def model_factory(v):
        model = _ruby_lattice_model(add_conjugate_hoppings=False)
        model.add_v(0, 1, [0, 0], v)
        model.add_v(3, 4, [0, 0], v)
        model.add_v(0, 3, [0, 0], v)
        return model

    k_points = uniform_k_mesh(2, 2)
    scan = scan_phase_diagram(
        model_factory,
        k_points,
        interaction_values=[0.0, 0.2],
        fillings=[0.4, 0.5],
        beta=8.0,
        max_iter=10,
        tol=1e-6,
    )

    assert scan["phase_labels"].shape == (2, 2)
    assert scan["charge_order"].shape == (2, 2)
    assert scan["loop_same"].shape == (2, 2)
    assert scan["loop_opposite"].shape == (2, 2)
    assert (0.2, 0.5) in scan["points"]
