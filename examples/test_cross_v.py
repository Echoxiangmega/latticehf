"""Test cross-cell V now fixed."""

import numpy as np
from latticehf import LatticeModel, KSpaceHFSolver

lat = np.array([[1.0, 0.0], [0.0, 1.0]])
model = LatticeModel(lat)
model.add_orbital([0.0, 0.0])
model.set_hop(-1.0, 0, 0, [1, 0])

print("=== V=0 ===")
k_path = np.array([[0.0, 0.0], [0.5, 0.0]])
solver = KSpaceHFSolver(model, k_path, nelectron=1)
solver.init_density_k()
solver.solve(max_iter=5, verbose=False)
print(f"E at Gamma: {solver.eigenvalues[0, 0]:.3f}")
print(f"E at X: {solver.eigenvalues[1, 0]:.3f}")

print("\n=== V=0.5, cross-cell ===")
model2 = LatticeModel(lat)
model2.add_orbital([0.0, 0.0])
model2.set_hop(-1.0, 0, 0, [1, 0])
model2.add_v(0, 0, [1, 0], 0.5)

solver2 = KSpaceHFSolver(model2, k_path, nelectron=1)
solver2.init_density_k()
solver2.solve(max_iter=20, verbose=False)
print(f"E at Gamma: {solver2.eigenvalues[0, 0]:.3f}")
print(f"E at X: {solver2.eigenvalues[1, 0]:.3f}")

print("\n=== V=0.5, same cell ===")
model3 = LatticeModel(lat)
model3.add_orbital([0.0, 0.0])
model3.set_hop(-1.0, 0, 0, [1, 0])
model3.add_v(0, 0, [0, 0], 0.5)

solver3 = KSpaceHFSolver(model3, k_path, nelectron=1)
solver3.init_density_k()
solver3.solve(max_iter=20, verbose=False)
print(f"E at Gamma: {solver3.eigenvalues[0, 0]:.3f}")
print(f"E at X: {solver3.eigenvalues[1, 0]:.3f}")