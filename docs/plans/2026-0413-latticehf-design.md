# LatticeHF Design

## Overview
Python包，用于处理二维晶格上考虑intersite库仑相互作用的周期体系，采用Hartree-Fock近似求解基态。

## Project Structure
```
latticehf/
├── __init__.py
├── model.py       # LatticeModel: 轨道、hopping、V定义
├── hamiltonian.py # Hamiltonian: 构建哈密顿量
├── solver.py      # HF求解器
└── quantities.py  # 物理量计算
```

## Core Components

### 1. LatticeModel
- 定义原胞、原子位置、轨道
- 定义hopping振幅 (t_ij)
- 定义轨道对相互作用 (V_ij)
- 支持有限超胞 + PBC

### 2. Hamiltonian
- 实空间构建
- 包含hopping + V的Hartree + Fock项
- 支持k空间变换

### 3. Solver
- SCF迭代求解HF基态
- k点采样
- 收敛判据

### 4. Quantities
- 电荷密度
- 能带结构
- 基态能量
- 从波函数可推导的其他量

## Algorithm
1. 初始猜测密度矩阵
2. 构建Fock矩阵 (H_tb + V_Hartree + V_Fock)
3. 对角化得到新轨道/本征值
4. 更新密度矩阵
5. 迭代至收敛