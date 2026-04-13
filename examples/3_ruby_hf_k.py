"""Example: Ruby lattice with k-space HF (V=0 vs V>0)."""

import numpy as np
import matplotlib.pyplot as plt
from latticehf import LatticeModel, KSpaceHFSolver

lat = np.array([[np.sqrt(3)/2,  1/2],
               [-np.sqrt(3)/2, 1/2]])

def make_model(V=0.0):
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
    
    if V > 0:
        model.add_v(0, 0, [0, 0], V)
        model.add_v(3, 3, [0, 0], V)
        model.add_v(0, 3, [0, 0], V)
    
    return model

rec = 2 * np.pi * np.linalg.inv(lat.T)
GAMMA = np.array([0.0, 0.0])
K_frac = np.array([1/3, 1/3])
M_frac = np.array([1/2, 0.0])

dist_GK = np.linalg.norm(K_frac @ rec - GAMMA @ rec)
dist_KM = np.linalg.norm(M_frac @ rec - K_frac @ rec)
dist_MG = np.linalg.norm(GAMMA @ rec - M_frac @ rec)

nk = 10
k_path = np.vstack([np.linspace(GAMMA, K_frac, nk),
                  np.linspace(K_frac, M_frac, nk),
                  np.linspace(M_frac, GAMMA, nk)])

dist_path = np.concatenate([
    np.linspace(0, dist_GK, nk),
    np.linspace(dist_GK, dist_GK + dist_KM, nk),
    np.linspace(dist_GK + dist_KM, dist_GK + dist_KM + dist_MG, nk)
])

nelectron = 4

print("=== V = 0 ===")
model0 = make_model(0)
solver0 = KSpaceHFSolver(model0, k_path, nelectron)
solver0.init_density_k()
solver0.solve(max_iter=20, tol=1e-6, verbose=False)
print(f"Ground state energy: {solver0.ground_state_energy:.4f}")

print("\n=== V = 0.3 ===")
model03 = make_model(0.3)
solver03 = KSpaceHFSolver(model03, k_path, nelectron)
solver03.init_density_k()
solver03.solve(max_iter=30, tol=1e-6)
print(f"Ground state energy: {solver03.ground_state_energy:.4f}")

band_v0 = solver0.get_band_structure()
band_v03 = solver03.get_band_structure()

plt.figure(figsize=(10, 6))

for b in range(6):
    plt.plot(dist_path, band_v0[:, b], 'b-', lw=1.5, label='V=0' if b==0 else None)

for b in range(6):
    plt.plot(dist_path, band_v03[:, b], 'r--', lw=1.5, label='V=0.3' if b==0 else None)

xticks = [0, dist_GK, dist_GK + dist_KM, dist_GK + dist_KM + dist_MG]
xlabels = ['Γ', 'K', 'M', 'Γ']
plt.xticks(xticks, xlabels)
plt.xlabel('k path')
plt.ylabel('Energy (eV)')
plt.title('Ruby Lattice: k-Space HF Band Structure')
plt.legend()
plt.axhline(0, color='k', ls='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ruby_hf_band.png', dpi=150)
plt.close()

print("\nSaved to ruby_hf_band.png")

print("\n=== Summary ===")
print(f"V=0:   E_gs = {solver0.ground_state_energy:.4f} eV/electron")
print(f"V=0.3: E_gs = {solver03.ground_state_energy:.4f} eV/electron")