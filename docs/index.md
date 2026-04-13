# LatticeHF Documentation

## Overview

LatticeHF is a Python package for computing ground states of 2D periodic lattice models with intersite Coulomb interactions using Hartree-Fock approximation.

## Core Concepts

### Lattice Model

Define a periodic lattice with:
- **Lattice vectors**: Real space primitive vectors
- **Orbitals**: Position of each orbital in unit cell
- **Hopping**: Nearest-neighbor hopping amplitudes
- **V interactions**: Intersite Coulomb interactions between specific orbital pairs

### Hartree-Fock Method

HF iteration:
1. Start with initial guess for density matrix
2. Build Fock matrix: F = H₀ + V_Hartree + V_Fock
3. Diagonalize to get new orbitals
4. Update density matrix
5. Repeat until convergence

## Module API

### model.LatticeModel

```python
from latticehf import LatticeModel

model = LatticeModel(lat_vecs)
```

**Methods:**

| Method | Description |
|--------|------------|
| `add_orbital(position, label)` | Add orbital at position (array) in unit cell |
| `set_hop(t, i, j, delta)` | Add hopping from orbital i to j with displacement delta |
| `add_v(i, j, delta, v)` | Add intersite V between orbitals i and j |
| `get_band_structure_k(k_path)` | Compute band structure along k-path |

**Arguments:**
- `position`: 2D array, orbital position in unit cell (e.g., `[0.0, 0.0]`)
- `delta`: 2D integer array, cell displacement (e.g., `[0,0]`, `[1,0]`, `[-1,1]`)
- `k_path`: shape (nk, 2), k-points in fractional coordinates

### hamiltonian.Hamiltonian

Real-space Hamiltonian with supercell:

```python
from latticehf import Hamiltonian

H = Hamiltonian(model, ncell=(Nx, Ny))
```

### solver.HFSolver

Real-space HF solver:

```python
from latticehf import HFSolver

solver = HFSolver(H, nelectron)
solver.init_density_guess()
solver.solve()
```

**Properties:**
- `ground_state_energy`: Total ground state energy
- `density_matrix`: Final density matrix
- `wavefunction`: Ground state wavefunction

### solver.KSpaceHFSolver

k-space HF solver for periodic systems:

```python
from latticehf import KSpaceHFSolver

solver = KSpaceHFSolver(model, k_path, nelectron)
solver.init_density_k()
solver.solve()
```

## Examples

### 1. Square Lattice (Real Space HF)

```python
import numpy as np
from latticehf import LatticeModel, Hamiltonian, HFSolver

# Square lattice
lat = np.array([[1.0, 0.0], [0.0, 1.0]])
model = LatticeModel(lat)
model.add_orbital([0.0, 0.0])

# Hopping
model.add_hopping(0, 0, [1, 0], -1.0)
model.add_hopping(0, 0, [0, 1], -1.0)

# Interaction
model.add_v(0, 0, [1, 0], 0.5)

# Solve
H = Hamiltonian(model, ncell=(4, 4))
solver = HFSolver(H, nelectron=8)
solver.init_density_guess()
solver.solve()

print(solver.ground_state_energy)
```

### 2. Ruby Lattice (k-Space HF)

```python
import numpy as np
from latticehf import LatticeModel, KSpaceHFSolver

# Ruby lattice vectors
lat = np.array([[np.sqrt(3)/2,  1/2],
               [-np.sqrt(3)/2, 1/2]])

model = LatticeModel(lat)

# 6 orbitals per unit cell
orb = [[1/3-1/8,    2/3+1/8],
       [1/3+2/8,  2/3+1/8],
       [1/3-1/8,    2/3-2/8],
       [2/3+1/8,    1/3-1/8],
       [2/3+1/8,    1/3+2/8],
       [2/3-2/8,    1/3-1/8]]
for pos in orb:
    model.add_orbital(pos)

# Hopping (ti: inside triangle, t1: between triangles)
ti, t1 = 0.4, 0.2
model.set_hop(ti, 0, 1, [0, 0])
model.set_hop(ti, 0, 2, [0, 0])
model.set_hop(ti, 2, 1, [0, 0])
model.set_hop(ti, 3, 4, [0, 0])
model.set_hop(ti, 3, 5, [0, 0])
model.set_hop(ti, 4, 5, [0, 0])
model.set_hop(t1, 1, 4, [0, 0])
model.set_hop(t1, 5, 0, [0, -1])
model.set_hop(t1, 2, 3, [-1, 0])
model.set_hop(t1, 3, 1, [0, -1])
model.set_hop(t1, 2, 5, [0, 0])
model.set_hop(t1, 0, 4, [-1, 0])

# Intersite V
model.add_v(0, 0, [0, 0], 0.3)
model.add_v(3, 3, [0, 0], 0.3)

# k-path: Γ-K-M-Γ in fractional coordinates
GAMMA = np.array([0.0, 0.0])
K_frac = np.array([1/3, 1/3])
M_frac = np.array([1/2, 0.0])

k_path = np.vstack([np.linspace(GAMMA, K_frac, 20),
                  np.linspace(K_frac, M_frac, 20),
                  np.linspace(M_frac, GAMMA, 20)])

# Solve HF
solver = KSpaceHFSolver(model, k_path, nelectron=4)
solver.init_density_k()
solver.solve()

print(f"Energy: {solver.ground_state_energy}")
```

### 3. Band Structure Plotting

```python
import numpy as np
import matplotlib.pyplot as plt
from latticehf import LatticeModel

lat = np.array([[np.sqrt(3)/2,  1/2],
               [-np.sqrt(3)/2, 1/2]])
model = LatticeModel(lat)
# ... add orbitals and hopping ...

# Reciprocal lattice
rec = 2 * np.pi * np.linalg.inv(lat.T)

# High-symmetry points (fractional)
GAMMA = np.array([0.0, 0.0])
K_frac = np.array([1/3, 1/3])
M_frac = np.array([1/2, 0.0])

# k-path distance in Cartesian
dist_GK = np.linalg.norm(K_frac @ rec - GAMMA @ rec)
dist_KM = np.linalg.norm(M_frac @ rec - K_frac @ rec)
dist_MG = np.linalg.norm(GAMMA @ rec - M_frac @ rec)

# Compute band structure
energies = model.get_band_structure_k(k_path)

# Plot
k_dist = np.concatenate([
    np.linspace(0, dist_GK, nk),
    np.linspace(dist_GK, dist_GK + dist_KM, nk),
    np.linspace(dist_GK + dist_KM, dist_GK + dist_KM + dist_MG, nk)
])

plt.figure()
for b in range(energies.shape[1]):
    plt.plot(k_dist, energies[:, b], 'b-')

plt.xticks([0, dist_GK, dist_GK + dist_KM, dist_GK + dist_KM + dist_MG], 
          ['Γ', 'K', 'M', 'Γ'])
plt.xlabel('k path')
plt.ylabel('Energy')
plt.savefig('band.png')
```

## Physical Background

### Hartree-Fock Approximation

The HF Hamiltonian:
$$F_{ij} = H_{ij} + \sum_kl V_{ikjl} \rho_{kl} - \sum_k V_{ikkj} \rho_{kj}$$

- **Hartree term**: $\sum_kl V_{ikjl} \rho_{kl}$ - classical electrostatic
- **Exchange term**: $-\sum_k V_{ikkj} \rho_{kj}$ - quantum

### Convergence

Set tolerance and max iterations:
```python
solver.solve(max_iter=100, tol=1e-6)
```

## References

- Based on Tight-Binding model similar to PythTB
- Hartree-Fock method for lattice models

## Advanced: Cross-Cell V Interactions

### Same Cell vs Cross Cell

V interactions can be defined between orbitals in the **same unit cell** or in **different unit cells**:

```python
# Same cell: V(i, j) in same cell
model.add_v(i, j, [0, 0], V)

# Cross cell: V(i, j) with cell at displacement delta
model.add_v(i, j, [1, 0], V)  # neighbor in x-direction
model.add_v(i, j, [0, 1], V)  # neighbor in y-direction
model.add_v(i, j, [1, 1], V)  # diagonal neighbor
```

### k-Space Treatment

In k-space, cross-cell V interactions include a **phase factor**:

$$V_k(i, j) = V \times e^{i \mathbf{k} \cdot \mathbf{R}_\delta}$$

Where $\mathbf{R}_\delta = \delta \cdot \mathbf{a}$ is the real-space displacement.

**Example**: 1D chain with V between nearest neighbors:

```python
lat = np.array([[1.0, 0.0], [0.0, 1.0]])
model = LatticeModel(lat)
model.add_orbital([0.0, 0.0])

# Hopping
model.set_hop(-1.0, 0, 0, [1, 0])

# Cross-cell V (neighbor sites)
model.add_v(0, 0, [1, 0], 0.5)

# k-path
GAMMA = np.array([0.0, 0.0])
X = np.array([0.5, 0.0])
k_path = np.vstack([GAMMA, X])

solver = KSpaceHFSolver(model, k_path, nelectron=1)
solver.solve()

print(f"Energy at Gamma: {solver.eigenvalues[0, 0]}")
print(f"Energy at X: {solver.eigenvalues[1, 0]}")
```

**Results**:
- At Γ (k=0): phase = 1, V contributes +0.5 to energy
- At X (k=0.5): phase = -1, V contributes -0.5 to energy

This demonstrates that cross-cell V correctly includes the k-dependent phase factor.