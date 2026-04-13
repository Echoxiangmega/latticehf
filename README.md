# LatticeHF

Hartree-Fock solver for lattice models with intersite Coulomb interactions in 2D periodic systems.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from latticehf import LatticeModel, Hamiltonian, HFSolver

# Define square lattice with one orbital per site
lat_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
model = LatticeModel(lat_vecs)
model.add_orbital([0.0, 0.0])

# Add nearest-neighbor hopping
model.add_hopping(0, 0, [1, 0], -1.0)
model.add_hopping(0, 0, [0, 1], -1.0)

# Add intersite Coulomb interaction
# Same cell: model.add_v(0, 0, [0, 0], 0.5)
# Cross cell: model.add_v(0, 0, [1, 0], 0.5)  # neighbor in x-direction
model.add_v(0, 0, [1, 0], 0.5)

# Solve HF in real space (supercell)
H = Hamiltonian(model, ncell=(4, 4))
solver = HFSolver(H, nelectron=8)
solver.init_density_guess()
solver.solve()

print(f"Ground state energy: {solver.ground_state_energy}")
```

## Modules

### 1. LatticeModel (`model.py`)
Define lattice structure, orbitals, hopping, and intersite V interactions.

**Key functions:**
- `add_orbital(position, label=None)` - Add orbital at position in unit cell
- `set_hop(t, i, j, delta)` - Add hopping term (alias for add_hopping)
- `add_v(i, j, delta, v)` - Add intersite Coulomb interaction
  - `delta=[0,0]`: same unit cell
  - `delta=[1,0]`, `[0,1]`: cross-cell (neighbor cells)
- `get_band_structure_k(k_path)` - Compute band structure in k-space

### 2. Hamiltonian (`hamiltonian.py`)
Build Hamiltonian with hopping + V (Hartree + Fock).

- `Hamiltonian(model, ncell)` - Real space Hamiltonian with supercell
- `KSpaceHamiltonian(model, k_points)` - k-space Hamiltonian

### 3. HFSolver (`solver.py`)
SCF iteration to solve Hartree-Fock equations.

- `HFSolver(hamiltonian, nelectron)` - Real space HF solver
- `KSpaceHFSolver(model, k_points, nelectron)` - k-space HF solver

### 4. Quantities (`quantities.py`)
Calculate physical observables from wavefunction.

- `charge_density()` - Charge density per site
- `structure_factor(q_points)` - Structure factor S(q)
- `momentum_distribution(k_points)` - Momentum distribution n(k)

## Advanced: Ruby Lattice Example

```python
import numpy as np
from latticehf import LatticeModel, KSpaceHFSolver

# Ruby lattice: 6 orbitals per unit cell
lat = np.array([[np.sqrt(3)/2,  1/2],
               [-np.sqrt(3)/2, 1/2]])

model = LatticeModel(lat)

# Add 6 orbitals per unit cell
orb = [[1/3-1/8,    2/3+1/8],
       [1/3+2/8,  2/3+1/8],
       [1/3-1/8,    2/3-2/8],
       [2/3+1/8,    1/3-1/8],
       [2/3+1/8,    1/3+2/8],
       [2/3-2/8,    1/3-1/8]]
for pos in orb:
    model.add_orbital(pos)

# t_i: inside triangles, t1: between triangles
ti, t1 = 0.4, 0.2

# Within triangles
model.set_hop(ti, 0, 1, [0, 0])
model.set_hop(ti, 0, 2, [0, 0])
model.set_hop(ti, 2, 1, [0, 0])

model.set_hop(ti, 3, 4, [0, 0])
model.set_hop(ti, 3, 5, [0, 0])
model.set_hop(ti, 4, 5, [0, 0])

# Between triangles
model.set_hop(t1, 1, 4, [0, 0])
model.set_hop(t1, 5, 0, [0, -1])
model.set_hop(t1, 2, 3, [-1, 0])

model.set_hop(t1, 3, 1, [0, -1])
model.set_hop(t1, 2, 5, [0, 0])
model.set_hop(t1, 0, 4, [-1, 0])

# Intersite V (same cell or cross-cell)
# model.add_v(0, 0, [0, 0], 0.3)  # same cell
model.add_v(0, 0, [1, 0], 0.3)  # cross-cell

# k-space HF
GAMMA = np.array([0.0, 0.0])
K_frac = np.array([1/3, 1/3])
M_frac = np.array([1/2, 0.0])

nk = 10
k_path = np.vstack([np.linspace(GAMMA, K_frac, nk),
                  np.linspace(K_frac, M_frac, nk),
                  np.linspace(M_frac, GAMMA, nk)])

solver = KSpaceHFSolver(model, k_path, nelectron=4)
solver.init_density_k()
solver.solve()

print(f"HF ground state energy: {solver.ground_state_energy}")
print(f"Band structure: {solver.get_band_structure()}")
```

## k-path for Band Structure

Use fractional (reduced) k-points for band structure plotting:

```python
rec = 2 * np.pi * np.linalg.inv(lat.T)

GAMMA = np.array([0.0, 0.0])
K_frac = np.array([1/3, 1/3])   # K point
M_frac = np.array([1/2, 0.0])   # M point

# Compute k-path distance for plotting
dist_GK = np.linalg.norm(K_frac @ rec - GAMMA @ rec)
dist_KM = np.linalg.norm(M_frac @ rec - K_frac @ rec)

k_path = np.vstack([np.linspace(GAMMA, K_frac, nk),
                  np.linspace(K_frac, M_frac, nk),
                  np.linspace(M_frac, GAMMA, nk)])

dist_path = np.concatenate([
    np.linspace(0, dist_GK, nk),
    np.linspace(dist_GK, dist_GK + dist_KM, nk),
    np.linspace(dist_GK + dist_KM, dist_GK + dist_KM + dist_MG, nk)
])

# Plotting
plt.plot(dist_path, energies)
plt.xticks([0, dist_GK, dist_GK + dist_KM], ['Γ', 'K', 'M'])
```


## Cross-Cell V Interactions

V interactions can be defined between orbitals in different unit cells:

```python
# Same cell: V(i, j) in same cell
model.add_v(i, j, [0, 0], V)

# Cross cell: V(i, j) with neighbor cell
model.add_v(i, j, [1, 0], V)  # x-direction
model.add_v(i, j, [0, 1], V)  # y-direction
```

In k-space, cross-cell V includes a phase factor:
$$V_k(i, j) = V \times e^{i \mathbf{k} \cdot \mathbf{R}_\delta}$$

**Example**: 1D chain with cross-cell V shows k-dependent energy shift.