"""
Quantities: 从基态波函数计算物理量。
"""

import numpy as np
from .model import LatticeModel


class Quantities:
    """
    计算各种物理量。
    """

    def __init__(self, model: LatticeModel, wavefunction: np.ndarray):
        """
        Args:
            model: LatticeModel实例
            wavefunction: 基态波函数，shape (nsite, nelectron)
        """
        self.model = model
        self.wavefunction = wavefunction
        self.nsite = model.nsite
        self.nelectron = wavefunction.shape[1]

    def charge_density(self):
        """
        计算每个格点的电荷密度。

        Returns:
            rho: shape (nsite,)
        """
        dm = self.wavefunction @ self.wavefunction.conj().T
        return np.real(np.diag(dm))

    def charge_density_per_orbital(self, iorb: int):
        """
        计算特定轨道的电荷密度。

        Args:
            iorb: 轨道索引

        Returns:
            rho_iorb: shape (ncell_total,)
        """
        norb = self.model.norb
        ncell = self.model.supercell_ncell

        rho = self.charge_density()
        rho_per_orb = rho[iorb::norb]

        return rho_per_orb

    def double_occupancy(self):
        """
        计算双占据数 (for Hubbard-like models)。

        Returns:
            double_occ: 每个格点的双占据数
        """
        dm = self.wavefunction @ self.wavefunction.conj().T
        return np.real(np.diag(dm) ** 2)

    def kinetic_energy(self, hopping: np.ndarray):
        """
        计算动能。

        Returns:
            T: 动能期望值
        """
        dm = self.wavefunction @ self.wavefunction.conj().T
        return np.sum(dm.conj() * hopping).real

    def potential_energy(self, v_pairs):
        """
        计算势能（intersite V贡献）。

        Returns:
            V: V相互作用的期望值
        """
        dm = self.wavefunction @ self.wavefunction.conj().T
        V = 0.0

        for (i, j, v) in v_pairs:
            V += v * dm[i, i].real * dm[j, j].real
            V -= v * np.abs(dm[i, j]) ** 2

        return V.real

    def spin_density(self, spin_up: np.ndarray, spin_down: np.ndarray):
        """
        计算自旋密度。

        Returns:
            sz: 每个格点的自旋密度 (n_z = n_up - n_down)
        """
        rho_up = spin_up @ spin_up.conj().T
        rho_down = spin_down @ spin_down.conj().T

        sz = np.real(np.diag(rho_up) - np.diag(rho_down))
        return sz

    def correlation_function(self, op_i: np.ndarray, op_j: np.ndarray):
        """
        计算关联函数 <A_i B_j>。

        Args:
            op_i, op_j: 算符在实空间的矩阵表示

        Returns:
            corr: 关联函数
        """
        dm = self.wavefunction @ self.wavefunction.conj().T
        return np.trace(op_i @ dm @ op_j @ dm)

    def structure_factor(self, q_points: np.ndarray):
        """
        计算结构因子 S(q) = <rho_q rho_{-q}>。

        Args:
            q_points: shape (nq, ndim)

        Returns:
            S_q: shape (nq,)
        """
        rho = self.charge_density()
        S_q = []

        for q in q_points:
            phase = np.exp(1j * q @ self.model.supercell_positions.T)
            rho_q = np.sum(phase * rho)
            S_q.append(np.abs(rho_q) ** 2)

        return np.array(S_q)

    def momentum_distribution(self, k_points: np.ndarray):
        """
        计算动量分布 n(k)。

        Args:
            k_points: shape (nk, ndim)

        Returns:
            n_k: shape (nk,)
        """
        n_k = []
        wf = self.wavefunction

        for k in k_points:
            phase = np.exp(1j * k @ self.model.supercell_positions.T)
            wf_k = phase @ wf
            n_k.append(np.sum(np.abs(wf_k) ** 2))

        return np.array(n_k)

    def fidelity(self, other_wavefunction: np.ndarray):
        """
        计算与另一个波函数的 fidelity。

        Args:
            other_wavefunction: 另一个基态波函数

        Returns:
            F: fidelity 值
        """
        overlap = self.wavefunction.conj().T @ other_wavefunction
        return np.abs(np.linalg.det(overlap))