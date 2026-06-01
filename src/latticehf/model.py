"""
LatticeModel: Define lattice, orbitals, hopping, and intersite interactions.
类似pythtb的方式定义周期性子体系。
"""

import numpy as np
from typing import Optional


class Orbital:
    """单个轨道定义。"""

    def __init__(self, position: np.ndarray, label: Optional[str] = None):
        self.position = np.asarray(position, dtype=float)
        self.label = label

    def __repr__(self):
        if self.label:
            return f"Orbital({self.position}, label={self.label})"
        return f"Orbital({self.position})"


class LatticeModel:
    """
    晶格模型，类似pythtb的tb_model。
    定义原胞、轨道、hopping和intersite V相互作用。
    """

    def __init__(self, lat_vecs: np.ndarray):
        """
        Args:
            lat_vecs: 原胞基矢，shape (2, 2) for 2D or (3, 3) for 3D
        """
        self.lat_vecs = np.asarray(lat_vecs, dtype=float)
        self.ndim = lat_vecs.shape[0]
        self.orbitals = []
        self.orbital_positions = []
        self.hopping_terms = []
        self.v_terms = []

    def add_orbital(self, position: np.ndarray, label: Optional[str] = None):
        """在原胞中添加一个轨道。"""
        position = np.asarray(position, dtype=float)
        orbital = Orbital(position, label)
        self.orbitals.append(orbital)
        self.orbital_positions.append(position)
        return len(self.orbitals) - 1

    def add_hopping(self, ind1: int, ind2: int, delta: np.ndarray, amplitude: complex):
        """
        添加hopping项。

        Args:
            ind1, ind2: 轨道索引（在原胞中）
            delta: 位移向量，用原胞基矢表示 (e.g., [0,0] = same cell, [1,0] = neighbor)
            amplitude: hopping振幅
        """
        self.hopping_terms.append({
            "i": ind1,
            "j": ind2,
            "delta": np.asarray(delta, dtype=int),
            "t": amplitude
        })

    def set_hop(self, amplitude: complex, ind1: int, ind2: int, delta: list):
        """Alias for add_hopping (to match pythtb API)."""
        self.add_hopping(ind1, ind2, delta, amplitude)

    def add_v(self, ind1: int, ind2: int, delta: np.ndarray, v: float):
        """
        添加intersite库仑相互作用项。

        Args:
            ind1, ind2: 轨道索引对（定义相互作用的两个轨道类型）
            delta: 位移向量
            v: 相互作用强度
        """
        self.v_terms.append({
            "i": ind1,
            "j": ind2,
            "delta": np.asarray(delta, dtype=int),
            "v": v
        })

    @property
    def norb(self):
        """原胞中的轨道数。"""
        return len(self.orbitals)

    def get_orbital_position(self, i: int) -> np.ndarray:
        """获取原胞中第i个轨道的位置。"""
        return np.asarray(self.orbital_positions[i])

    def make_supercell(self, ncell: tuple):
        """
        创建超胞。

        Args:
            ncell: 每个方向的超胞大小，如 (4, 4)

        Returns:
            supercell_positions: 所有轨道位置
            supercell_indices: (orb_index, cell_i, cell_j) -> global_index
        """
        nx, ny = ncell
        positions = []
        indices = {}

        for ix in range(nx):
            for iy in range(ny):
                for iorb in range(self.norb):
                    r = (np.array([ix, iy]) + self.orbital_positions[iorb]) @ self.lat_vecs
                    positions.append(r)
                    indices[(iorb, ix, iy)] = len(positions) - 1

        return np.array(positions), indices

    def set_periodic_hopping(self, ncell: tuple):
        """将hopping和V项扩展到超胞（周期边界）。"""
        self.supercell_ncell = ncell
        self.supercell_positions, self.supercell_indices = self.make_supercell(ncell)
        self.nsite = len(self.supercell_positions)

    def get_hopping_matrix(self, ncell: tuple = None) -> np.ndarray:
        """构建实空间hopping矩阵。"""
        if ncell is not None:
            self.set_periodic_hopping(ncell)

        n = self.nsite
        H = np.zeros((n, n), dtype=complex)

        for term in self.hopping_terms:
            iorb, jorb = term["i"], term["j"]
            delta = term["delta"]
            t = term["t"]

            for ix in range(self.supercell_ncell[0]):
                for iy in range(self.supercell_ncell[1]):
                    jx = (ix + delta[0]) % self.supercell_ncell[0]
                    jy = (iy + delta[1]) % self.supercell_ncell[1]

                    if (iorb, ix, iy) in self.supercell_indices and \
                       (jorb, jx, jy) in self.supercell_indices:
                        gi = self.supercell_indices[(iorb, ix, iy)]
                        gj = self.supercell_indices[(jorb, jx, jy)]
                        H[gi, gj] += t

        return H

    def get_v_pairs(self, ncell: tuple = None):
        """获取V相互作用的所有轨道对。"""
        if ncell is not None:
            self.set_periodic_hopping(ncell)

        pairs = []
        for term in self.v_terms:
            iorb, jorb = term["i"], term["j"]
            delta = term["delta"]
            v = term["v"]

            for ix in range(self.supercell_ncell[0]):
                for iy in range(self.supercell_ncell[1]):
                    jx = (ix + delta[0]) % self.supercell_ncell[0]
                    jy = (iy + delta[1]) % self.supercell_ncell[1]

                    if (iorb, ix, iy) in self.supercell_indices and \
                       (jorb, jx, jy) in self.supercell_indices:
                        gi = self.supercell_indices[(iorb, ix, iy)]
                        gj = self.supercell_indices[(jorb, jx, jy)]
                        pairs.append((gi, gj, v))

        return pairs

    def get_band_structure_k(self, k_path: np.ndarray) -> np.ndarray:
        """
        计算k空间能带结构。

        Args:
            k_path: shape (nk, ndim), k点路径（倒格子分数坐标，如[0,0], [1/3,1/3]）

        Returns:
            energies: shape (nk, nband)
        """
        nk = len(k_path)
        nband = self.norb
        energies = np.zeros((nk, nband))

        lat_vecs = self.lat_vecs
        rec_vecs = 2 * np.pi * np.linalg.inv(lat_vecs.T)

        for ik, k_frac in enumerate(k_path):
            k_cart = k_frac @ rec_vecs

            H_k = np.zeros((nband, nband), dtype=complex)

            for term in self.hopping_terms:
                iorb, jorb = term["i"], term["j"]
                delta = term["delta"]
                t = term["t"]

                r_i = self.orbital_positions[iorb] @ lat_vecs
                r_j = self.orbital_positions[jorb] @ lat_vecs
                delta_R = delta @ lat_vecs
                dr = r_j - r_i + delta_R

                phase = np.exp(1j * np.dot(k_cart, dr))
                t_k = t * phase

                H_k[iorb, jorb] += t_k
                H_k[jorb, iorb] += t_k.conj()

            e, _ = np.linalg.eigh(H_k)
            energies[ik, :] = e

        return energies
