# Green-Function View of LatticeHF

This note explains how the current Hartree-Fock implementation can be read in
Green-function language, and how the same objects can be extended toward more
general interacting approximations for periodic tight-binding models.

## Model

For a periodic tight-binding model with orbitals inside a unit cell,

```text
H = sum_{ij} h^0_{ij} c_i^dagger c_j
  + 1/2 sum_{ij} V_{ij} n_i n_j,
```

where `i,j` may denote real-space supercell sites or orbital indices with cell
translations.  In the present code, `LatticeModel.hopping_terms` defines `h0`,
and `v_terms` defines density-density interactions between selected sites or
orbitals.

## Hartree-Fock as a Static Self-Energy

HF is a Dyson equation with a static density-dependent self-energy:

```text
G^{-1}(i omega_n) = (i omega_n + mu) I - h0 - Sigma_HF[rho]
rho_ij = <c_j^dagger c_i> = G_ij(tau=0^-)
```

For density-density interactions,

```text
(Sigma_H)_ii += V_ij rho_jj
(Sigma_H)_jj += V_ij rho_ii
(Sigma_F)_ij -= V_ij rho_ji
(Sigma_F)_ji -= V_ij rho_ij
```

Thus the current SCF iteration is the fixed-point equation

```text
rho = f(h0 + Sigma_HF[rho]),
```

where `f` is the zero-temperature projector at fixed electron number, or the
finite-temperature Fermi function at fixed chemical potential.  The helper module
`latticehf.green` exposes this explicitly through:

- `hartree_fock_self_energy`
- `matsubara_green`
- `zero_temperature_density_matrix`
- `finite_temperature_density_matrix`
- `green_density_residual`

This representation is useful because HF, GW, and post-GW can share the same
Dyson-equation interface while differing only in the self-energy and screened
interaction.

## Periodic and Finite Geometries

Two complementary workflows should be kept:

1. Periodic primitive-cell calculations using a k mesh.  This is the natural
   setup for phase diagrams and order parameters at crystal momenta.
2. Repeated finite supercells with either periodic or open boundaries.  This is
   useful for commensurate charge order, loop-current patterns, and comparison
   with exact diagonalization or finite-cluster methods.

For a symmetry-broken phase with period larger than the primitive cell, use a
commensurate supercell and treat its internal orbitals as the enlarged unit cell.
Then the same Green-function language applies with `h0(k)` and `Sigma(k)` in the
folded Brillouin zone.

## Order Parameters

All one-body HF order parameters can be expressed from the density matrix
`rho_ij = <c_j^dagger c_i>`.

### Charge Order

Site charge:

```text
n_i = rho_ii
```

A charge-density-wave component at wave vector q is

```text
n(q) = sum_i exp(i q dot r_i) (n_i - n_bar).
```

Use `charge_order_amplitude` for a real-space pattern and
`charge_structure_factor` for q-space peaks.

### Bond Order

Bond order on an oriented bond `i <- j`:

```text
B_ij = Re <c_i^dagger c_j> = Re rho_ji.
```

This diagnoses dimerization or orbital/bond nematicity even when site charges are
uniform.

### Loop Current

For a one-body Hamiltonian

```text
H0 = sum_ij h_ij c_i^dagger c_j,
```

the continuity-equation convention gives the oriented bond current

```text
J_{i -> j} = 2 Im[ h_ij <c_i^dagger c_j> ] = 2 Im[ h_ij rho_ji ].
```

A loop-current order parameter is the oriented average around a plaquette or
triangle:

```text
J_loop = mean_{(i -> j) in loop} J_{i -> j}.
```

For the Ruby lattice, define the small triangles and larger plaquettes as
oriented loops.  Then ferro-loop-current and staggered-loop-current phases are
obtained by projecting loop currents onto a sign pattern with
`loop_current_pattern`.

## Toward GW and Post-GW

Standard GW replaces the static HF self-energy by

```text
Sigma_GW(k, i omega_n) = - sum_{q, i Omega_m} G(k+q, i omega_n+i Omega_m) W(q, i Omega_m),
W_RPA(q, i Omega_m) = [1 - V(q) P(q, i Omega_m)]^{-1} V(q),
P(q, i Omega_m) = sum_{k, i omega_n} G(k+q, i omega_n+i Omega_m) G(k, i omega_n).
```

The post-GW paper changes the screened interaction used in the one-shot Green
function.  Instead of using the RPA response, it computes a covariant physical
response `chi_cov` and reconnects the screened interaction to it:

```text
W_post(q, i Omega_m) = V(q) - V(q) chi_cov(q, i Omega_m) V(q),
G_post^{-1} = G0_H^{-1} - Sigma_post,
Sigma_post = - G_tr W_post.
```

For this package, a practical roadmap is:

1. Keep `rho`, `G(iw)`, and `Sigma` as first-class objects.
2. Implement `InteractionKernel` objects that can produce `V(q)` from `v_terms`.
3. Implement bubble response `P0(q, iOmega)` as the first non-HF response.
4. Add RPA/GW `W` and one-shot `Sigma_GW` on top of the existing k-mesh code.
5. Add a covariant-response interface `chi_cov(q, iOmega)`; the full cGW vertex
   is the hard part and should be isolated behind this interface.
6. Compute `W_post` and `G_post` one-shot, then reuse the order-parameter module
   to compare charge order, bond order, and loop-current tendencies.

The important architectural point is that observables should not depend on
whether `rho` came from HF, GW, post-GW, finite clusters, or exact diagonalization.
They should consume Green functions or density matrices only.
