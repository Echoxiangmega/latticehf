# Periodic Finite-Temperature HF Phase Scans

`latticehf.phase_diagram` provides a small workflow for translationally
invariant finite-temperature HF phase diagrams on a `(V, n)` grid.

Here `V` is supplied through a user-defined `model_factory(V)`, and `n` is the
filling.  By default, `filling_unit="fraction"`, so `n=0.5` means half filling
of the orbitals in the unit cell.  Use `filling_unit="particles"` if `n` should
mean particles per unit cell directly.

```python
import numpy as np
from latticehf import LatticeModel, scan_phase_diagram, uniform_k_mesh


def ruby_model(V):
    model = LatticeModel(...)
    # add orbitals and hoppings
    model.add_v(0, 1, [0, 0], V)
    model.add_v(3, 4, [0, 0], V)
    model.add_v(0, 3, [0, 0], V)
    return model


k_points = uniform_k_mesh(12, 12)
V_values = np.linspace(0.0, 1.0, 21)
fillings = np.linspace(0.2, 0.8, 25)

scan = scan_phase_diagram(
    ruby_model,
    k_points,
    interaction_values=V_values,
    fillings=fillings,
    beta=20.0,
)
```

The returned dictionary contains arrays with shape `(len(V_values),
len(fillings))`:

- `phase_labels`
- `charge_order`
- `loop_same`
- `loop_opposite`

For the Ruby lattice, the default diagnostics use the two intra-cell triangles:

```text
upper triangle: 0 -> 1 -> 2 -> 0
lower triangle: 3 -> 4 -> 5 -> 3
```

The two loop-current patterns are

```text
loop_same     = (J_upper + J_lower) / 2
loop_opposite = (J_upper - J_lower) / 2
```

The raw point data are available as

```python
point = scan["points"][(V, n)]
result = point["result"]
orders = point["orders"]
phase = point["phase"]
```

`result.density_matrix` is the averaged unit-cell density matrix.  `result.density_k`
keeps the k-resolved density matrices, which is useful for later GW/post-GW
extensions.

## Notes

Use a uniform, time-reversal-symmetric k mesh for loop-current scans.  Small or
asymmetric k sets can produce finite-size current-like artifacts.

The current implementation follows the repository's existing k-space HF
convention: the interaction self-energy is built from a k-averaged unit-cell
density matrix.  A more complete periodic Fock term should eventually be written
as a momentum convolution; the module is structured so that replacement can
happen behind the same `Sigma(k)` interface.
