"""
Hamiltonian: 构建包含hopping和V（Hartree+Fock）的哈密顿量。
"""

import numpy as np
from .model import LatticeModel


class Hamiltonian:
    """
    哈密顿量构建器。
    H = H_hopping + V_Hartree + V_Fock
    """

    def __init__(self, model: LatticeModel, ncell: tuple):
        """
        Args:
            model: LatticeModel实例
            ncell: 超胞大小
        """
        self.model = model
        self.ncell = ncell
        self.model.set_periodic_hopping(ncell)
        self.nsite = model.nsite

        self.H_hopping = model.get_hopping_matrix()
        self.v_pairs = model.get_v_pairs()

        self.density_matrix = None
        self.fock_matrix = None

    def set_density_matrix(self, dm):
        """设置密度矩阵。"""
        self.density_matrix = np.asarray(dm, dtype=complex)

    def compute_fock(self, density_matrix: np.ndarray = None) -> np.ndarray:
        """
        构建Fock矩阵。

        F = H_0 + V_Hartree + V_Fock

        Hartree项: V_H[i,j] = sum_kl V[i,k,j,l] * rho[k,l]
        交换项:   V_X[i,j] = -sum_k V[i,k,j,k] * rho[k,j]

        Args:
            density_matrix: 密度矩阵 rho
        """
        if density_matrix is not None:
            self.density_matrix = density_matrix

        if self.density_matrix is None:
            raise ValueError("Density matrix not set. Set initial guess first.")

        F = self.H_hopping.copy()

        for (i, j, v) in self.v_pairs:
            rho_ii = self.density_matrix[i, i].real
            rho_jj = self.density_matrix[j, j].real
            rho_ij = self.density_matrix[i, j]
            rho_ji = self.density_matrix[j, i]

            F[i, i] += v * rho_jj
            F[j, j] += v * rho_ii

            F[i, j] -= v * rho_ji
            F[j, i] -= v * rho_ij

        self.fock_matrix = F
        return F

    def get_hamiltonian(self, density_matrix: np.ndarray = None) -> np.ndarray:
        """获取完整哈密顿量。"""
        if density_matrix is not None:
            return self.compute_fock(density_matrix)
        elif self.fock_matrix is not None:
            return self.fock_matrix
        else:
            return self.H_hopping

    def compute_energy(self, density_matrix: np.ndarray = None) -> float:
        """
        计算基态能量。

        E = Tr(rho * H) + 1/2 * sum_ij V_ij * rho_ii * rho_jj
           - 1/2 * sum_ij V_ij * rho_ij^2
        """
        if density_matrix is not None:
            dm = density_matrix
        else:
            dm = self.density_matrix

        if dm is None:
            raise ValueError("Density matrix not set.")

        e1 = np.sum(dm.conj() * self.H_hopping).real

        e2 = 0.0
        e3 = 0.0
        for (i, j, v) in self.v_pairs:
            e2 += 0.5 * v * dm[i, i].real * dm[j, j].real
            e3 -= 0.5 * v * np.abs(dm[i, j]) ** 2

        return (e1 + e2 + e3).real


class KSpaceHamiltonian:
    """
    k空间哈密顿量，用于周期体系的k点计算。
    """

    def __init__(self, model: LatticeModel, k_points: np.ndarray, gauge: str = "embedding"):
        """
        Args:
            model: LatticeModel实例
            k_points: k点路径，shape (nk, ndim)
        """
        if gauge not in {"embedding", "cell"}:
            raise ValueError("gauge must be 'embedding' or 'cell'")

        self.model = model
        self.k_points = k_points
        self.nk = len(k_points)
        self.norb = model.norb
        self.gauge = gauge

        self.H_k_cache = {}
        self._cache_hopping()

    def _cache_hopping(self):
        """缓存每个k点的hopping矩阵。"""
        lat_vecs = self.model.lat_vecs

        for ik, k_frac in enumerate(self.k_points):
            rec_vecs = 2 * np.pi * np.linalg.inv(lat_vecs.T)
            k_cart = k_frac @ rec_vecs
            H_k = np.zeros((self.norb, self.norb), dtype=complex)

            for term in self.model.hopping_terms:
                iorb, jorb = term["i"], term["j"]
                delta = term["delta"]
                t = term["t"]

                delta_R = delta @ lat_vecs
                if self.gauge == "embedding":
                    r_i = self.model.orbital_positions[iorb] @ lat_vecs
                    r_j = self.model.orbital_positions[jorb] @ lat_vecs
                    dr = r_j - r_i + delta_R
                else:
                    dr = delta_R
                phase = np.exp(1j * np.dot(k_cart, dr))
                t_k = t * phase

                H_k[iorb, jorb] += t_k
                H_k[jorb, iorb] += t_k.conj()

            self.H_k_cache[ik] = H_k

    def get_hopping(self, ik: int) -> np.ndarray:
        """获取指定k点的hopping矩阵。"""
        return self.H_k_cache[ik]

    def add_v_fock_k(self, density_k: np.ndarray, v_term: dict, ik: int) -> np.ndarray:
        """
        在k空间添加V的Hartree和Fock项。

        Args:
            density_k: k空间的密度矩阵
            v_term: {'i': iorb, 'j': jorb, 'delta': delta, 'v': v}
            ik: k点索引

        Returns:
            V_k: V贡献的矩阵
        """
        iorb = v_term['i']
        jorb = v_term['j']
        v = v_term['v']
        delta = v_term['delta']

        k_frac = self.k_points[ik]
        rec_vecs = 2 * np.pi * np.linalg.inv(self.model.lat_vecs.T)
        k_cart = k_frac @ rec_vecs

        delta_R = delta @ self.model.lat_vecs
        phase = np.exp(1j * np.dot(k_cart, delta_R))

        v_eff = v * phase

        V_vk = np.zeros((self.norb, self.norb), dtype=complex)

        rho_ii = density_k[iorb, iorb].real
        rho_jj = density_k[jorb, jorb].real
        rho_ij = density_k[iorb, jorb]
        rho_ji = density_k[jorb, iorb]

        Hartree_ii = v_eff * rho_jj
        Hartree_jj = v_eff * rho_ii
        Exchange_ij = v_eff * rho_ji
        Exchange_ji = v_eff.conj() * rho_ij

        V_vk[iorb, iorb] = Hartree_ii
        V_vk[jorb, jorb] = Hartree_jj
        V_vk[iorb, jorb] = -Exchange_ij
        V_vk[jorb, iorb] = -Exchange_ji

        return V_vk
