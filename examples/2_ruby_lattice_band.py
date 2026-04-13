"""
Ruby lattice band structure with proper k-path distance for plotting.
参考pythtb的做法。
"""

import numpy as np
import matplotlib.pyplot as plt
from latticehf import LatticeModel

lat = np.array([[np.sqrt(3)/2,  1/2],
               [-np.sqrt(3)/2, 1/2]])

rec = 2 * np.pi * np.linalg.inv(lat.T)

model = LatticeModel(lat)

orb = [[1/3-1/8,    2/3+1/8],
       [1/3+2/8,  2/3+1/8],
       [1/3-1/8,    2/3-2/8],
       [2/3+1/8,    1/3-1/8],
       [2/3+1/8,    1/3+2/8],
       [2/3-2/8,    1/3-1/8]]
for pos in orb:
    model.add_orbital(pos)

ti = 0.4
t1 = 0.2

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

nk = 30

GAMMA = np.array([0.0, 0.0])
K_frac = np.array([1/3, 1/3])
M_frac = np.array([1/2, 0.0])

dist_GK = np.linalg.norm(K_frac @ rec - GAMMA @ rec)
dist_KM = np.linalg.norm(M_frac @ rec - K_frac @ rec)
dist_MG = np.linalg.norm(GAMMA @ rec - M_frac @ rec)

print(f"Distances: Γ-K = {dist_GK:.3f}, K-M = {dist_KM:.3f}, M-Γ = {dist_MG:.3f}")

k_path = np.vstack([np.linspace(GAMMA, K_frac, nk),
                  np.linspace(K_frac, M_frac, nk),
                  np.linspace(M_frac, GAMMA, nk)])

dist_path = np.concatenate([
    np.linspace(0, dist_GK, nk),
    np.linspace(dist_GK, dist_GK + dist_KM, nk),
    np.linspace(dist_GK + dist_KM, dist_GK + dist_KM + dist_MG, nk)
])

energies = model.get_band_structure_k(k_path)

plt.figure(figsize=(10, 6))
for band in range(energies.shape[1]):
    plt.plot(dist_path, energies[:, band], 'b-', lw=1.5)

xticks = [0, dist_GK, dist_GK + dist_KM, dist_GK + dist_KM + dist_MG]
xlabels = ['Γ', 'K', 'M', 'Γ']
plt.xticks(xticks, xlabels)
plt.xlabel('k path')
plt.ylabel('Energy (eV)')
plt.title(f'Ruby Lattice Band (ti={ti}, t1={t1})')
plt.axhline(0, color='k', ls='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ruby_band.png', dpi=150)
plt.close()

print(f"\nTotal k-path length: {dist_GK + dist_KM + dist_MG:.3f}")
print("Saved to ruby_band.png")