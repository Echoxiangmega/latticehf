"""Debug same cell V in Ruby lattice."""

import numpy as np
from latticehf import LatticeModel, KSpaceHFSolver

lat = np.array([[np.sqrt(3)/2, 1/2], [-np.sqrt(3)/2, 1/2]])
model = LatticeModel(lat)

orb = [[1/3-1/8, 2/3+1/8], [1/3+2/8, 2/3+1/8], [1/3-1/8, 2/3-2/8],
       [2/3+1/8, 1/3-1/8], [2/3+1/8, 1/3+2/8], [2/3-2/8, 1/3-1/8]]
for pos in orb:
    model.add_orbital(pos)

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

GAMMA = np.array([0.0, 0.0])
k_path = np.array([GAMMA])

# Test V=0
solver0 = KSpaceHFSolver(model, k_path, nelectron=4)
solver0.init_density_k()
solver0.solve(max_iter=5, verbose=False)
print(f"V=0, total E at Gamma: {solver0.eigenvalues[0].sum():.4f}")
print(f"  Bands: {solver0.eigenvalues[0]}")

# Add same-cell V
model2 = LatticeModel(lat)
for pos in orb:
    model2.add_orbital(pos)
for i,j,d in [(0,1,[0,0]),(0,2,[0,0]),(2,1,[0,0]),(3,4,[0,0]),(3,5,[0,0]),(4,5,[0,0]),(1,4,[0,0]),(5,0,[0,-1]),(2,3,[-1,0]),(3,1,[0,-1]),(2,5,[0,0]),(0,4,[-1,0])]:
    model2.set_hop(ti if i<3 else t1, i, j, d)
model2.add_v(0, 0, [0, 0], 0.3)
model2.add_v(3, 3, [0, 0], 0.3)

solver1 = KSpaceHFSolver(model2, k_path, nelectron=4)
solver1.init_density_k()
solver1.solve(max_iter=30, verbose=False)
print(f"\nV=0.3(same cell), total E at Gamma: {solver1.eigenvalues[0].sum():.4f}")
print(f"  Bands: {solver1.eigenvalues[0]}")