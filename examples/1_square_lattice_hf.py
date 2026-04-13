"""
Example: Square lattice with nearest-neighbor hopping and intersite V.
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from latticehf import LatticeModel, Hamiltonian, HFSolver, Quantities

lat_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
model = LatticeModel(lat_vecs)

model.add_orbital([0.0, 0.0], label="s")

model.add_hopping(0, 0, [1, 0], -1.0)
model.add_hopping(0, 0, [0, 1], -1.0)

model.add_v(0, 0, [1, 0], 0.5)
model.add_v(0, 0, [0, 1], 0.5)

ncell = (4, 4)
model.set_periodic_hopping(ncell)
print(f"Supercell: {ncell}")
print(f"Number of sites: {model.nsite}")

H = Hamiltonian(model, ncell)
nelectron = model.nsite // 2
solver = HFSolver(H, nelectron)

solver.init_density_guess("atomic")
converged, energies = solver.solve(max_iter=50, tol=1e-6)

if converged:
    print(f"\nGround state energy: {solver.ground_state_energy:.6f}")
    print(f"Orbital energies: {solver.get_orbital_energies()}")

    wf = solver.get_wavefunction()
    quants = Quantities(model, wf)
    rho = quants.charge_density()
    print(f"\nCharge density (first 4 sites): {rho[:4]}")
else:
    print("HF did not converge")