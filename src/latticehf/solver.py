"""
HFSolver: Hartree-Fock自洽场迭代求解基态。
同时支持实空间和k空间求解。
"""

import numpy as np
from .hamiltonian import Hamiltonian, KSpaceHamiltonian


class HFSolver:
    """
    Hartree-Fock求解器。
    通过SCF迭代求解HF基态。
    """

    def __init__(self, hamiltonian: Hamiltonian, nelectron: int, nk: int = 1):
        """
        Args:
            hamiltonian: Hamiltonian实例
            nelectron: 电子数
            nk: k点数（用于k空间采样）
        """
        self.hamiltonian = hamiltonian
        self.nelectron = nelectron
        self.nk = nk
        self.nsite = hamiltonian.nsite

        self.eigenvalues = None
        self.eigenvectors = None
        self.density_matrix = None
        self.ground_state_energy = None

    def init_density_guess(self, mode: str = "atomic"):
        """
        初始化密度矩阵猜测。

        Args:
            mode: "atomic" (每个轨道一个电子) 或 "random"
        """
        if mode == "atomic":
            dm = np.zeros((self.nsite, self.nsite), dtype=complex)
            for i in range(self.nelectron):
                dm[i, i] = 1.0
        elif mode == "random":
            dm = np.random.random((self.nsite, self.nsite)) * 0.1
            dm = dm @ dm.T
            dm = dm / np.trace(dm) * self.nelectron
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.density_matrix = dm
        return dm

    def solve(self, max_iter: int = 100, tol: float = 1e-6, verbose: bool = True):
        """
        运行HF自洽迭代。

        Args:
            max_iter: 最大迭代次数
            tol: 收敛判据 (||rho_new - rho_old||_max)
            verbose: 是否打印迭代信息

        Returns:
            converged: 是否收敛
            energies: 每次迭代的能量
        """
        energies = []

        for iteration in range(max_iter):
            F = self.hamiltonian.compute_fock(self.density_matrix)

            eigenvalues, eigenvectors = np.linalg.eigh(F)

            idx = np.argsort(eigenvalues)
            self.eigenvalues = eigenvalues[idx]
            self.eigenvectors = eigenvectors[:, idx]

            occupied = self.eigenvectors[:, :self.nelectron]
            rho_new = occupied @ occupied.conj().T

            diff = np.max(np.abs(rho_new - self.density_matrix))
            self.density_matrix = rho_new

            energy = self.hamiltonian.compute_energy(rho_new)
            energies.append(energy)

            if verbose:
                print(f"Iter {iteration + 1}: E = {energy:.6f}, diff = {diff:.2e}")

            if diff < tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        else:
            if verbose:
                print("Did not converge within max_iter")

        self.ground_state_energy = energies[-1] if energies else None
        return len(energies) < max_iter, energies

    def get_wavefunction(self):
        """获取基态波函数。"""
        return self.eigenvectors[:, :self.nelectron]

    def get_density_matrix(self):
        """获取基态密度矩阵。"""
        return self.density_matrix

    def get_orbital_energies(self):
        """获取占据轨道能量。"""
        return self.eigenvalues[:self.nelectron]

    def get_band_structure(self, k_points: np.ndarray):
        """
        计算能带结构（k空间）。

        Args:
            k_points: shape (nk, ndim)

        Returns:
            energies: shape (nk, nband)
        """
        H0 = self.hamiltonian.H_hopping
        model = self.hamiltonian.model
        nk = len(k_points)
        nband = self.nsite

        energies = np.zeros((nk, nband))
        positions = model.supercell_positions

        for ik, k in enumerate(k_points):
            phase = np.exp(1j * k @ positions.T)
            phase_matrix = phase[:, np.newaxis] * phase[np.newaxis, :].conj()
            H_k = H0 * phase_matrix
            e, _ = np.linalg.eigh(H_k)
            energies[ik, :] = e

        return energies


class KSpaceHFSolver:
    """
    k空间HF求解器。
    在每个k点独立做HF自洽，然后处理k空间密度矩阵。
    """

    def __init__(self, model, k_points: np.ndarray, nelectron: int):
        """
        Args:
            model: LatticeModel实例
            k_points: k点路径
            nelectron: 总电子数
        """
        self.model = model
        self.k_points = k_points
        self.nk = len(k_points)
        self.norb = model.norb
        self.nelectron = nelectron

        self.hamiltonian_k = KSpaceHamiltonian(model, k_points)

        self.density_k = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.ground_state_energy = None

    def init_density_k(self, mode: str = "atomic"):
        """初始化k空间密度矩阵猜测。"""
        nocc_per_k = max(1, self.nelectron // self.nk)
        dm = np.zeros((self.norb, self.norb), dtype=complex)
        for i in range(min(nocc_per_k, self.norb)):
            dm[i, i] = 1.0
        self.density_k = dm
        return dm

    def solve(self, max_iter: int = 100, tol: float = 1e-6, verbose: bool = True):
        """运行k空间HF自洽迭代。"""
        energies = []

        for iteration in range(max_iter):
            F_k = np.zeros((self.nk, self.norb, self.norb), dtype=complex)
            eigenvalues = np.zeros((self.nk, self.norb))
            eigenvectors = np.zeros((self.nk, self.norb, self.norb), dtype=complex)

            nocc_per_k = max(1, self.nelectron // self.nk)

            for ik in range(self.nk):
                H0 = self.hamiltonian_k.get_hopping(ik)
                F_k[ik] = H0

                if self.density_k is not None:
                    for v_term in self.model.v_terms:
                        V_k = self.hamiltonian_k.add_v_fock_k(self.density_k, v_term, ik)
                        F_k[ik] += V_k

                e, v = np.linalg.eigh(F_k[ik])
                eigenvalues[ik] = e
                eigenvectors[ik] = v

            rho_new = np.zeros((self.norb, self.norb), dtype=complex)

            for ik in range(self.nk):
                occupied = eigenvectors[ik, :, :nocc_per_k]
                rho_new += occupied @ occupied.conj().T

            rho_new /= self.nk
            diff = np.max(np.abs(rho_new - self.density_k))
            self.density_k = rho_new

            e_total = 0.0
            for ik in range(self.nk):
                e = eigenvalues[ik, :nocc_per_k].sum()
                e_total += e
            e_total /= self.nk

            energies.append(e_total)

            if verbose:
                print(f"Iter {iteration + 1}: E = {e_total:.6f}, diff = {diff:.2e}")

            if diff < tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.ground_state_energy = energies[-1] if energies else None
        return len(energies) < max_iter, energies

    def get_band_structure(self):
        """获取HF��带。"""
        return self.eigenvalues