# 量子蒙地卡羅數值模擬專案 (Quantum Monte Carlo Simulation)

本專案實現了量子蒙地卡羅（Quantum Monte Carlo, QMC）的最新技術進展，用於研究強關聯量子多體系統、量子相變和超固體相變。

## 專案特色

### 核心技術
- **神經網路量子態 (Neural Network Quantum States, NQS)**: 使用深度神經網路作為變分波函數
- **變分蒙地卡羅 (Variational Monte Carlo, VMC)**: 結合機器學習優化的變分方法
- **輔助場QMC (Auxiliary-Field QMC, AFQMC)**: 用於費米子系統的格點模型
- **路徑積分蒙地卡羅 (Path Integral Monte Carlo, PIMC)**: 用於玻色子系統和相變研究

### 應用模型
- **Hubbard模型**: 強關聯電子系統
- **Bose-Hubbard模型**: 超流-絕緣體相變
- **超固體相變**: 各種晶格結構中的超固體研究

## 專案結構

```
qmc/
├── qmc/                    # 核心模組
│   ├── __init__.py
│   ├── core/              # 核心框架
│   │   ├── __init__.py
│   │   ├── base.py        # 基礎類別
│   │   ├── lattice.py     # 晶格結構
│   │   └── utils.py       # 工具函數
│   ├── nqs/               # 神經網路量子態
│   │   ├── __init__.py
│   │   ├── rbm.py         # 受限玻爾茲曼機
│   │   ├── cnn.py         # 卷積神經網路量子態
│   │   └── mlp.py         # 多層感知器量子態
│   ├── methods/           # QMC方法
│   │   ├── __init__.py
│   │   ├── vmc.py         # 變分蒙地卡羅
│   │   ├── afqmc.py       # 輔助場QMC
│   │   └── pimc.py        # 路徑積分蒙地卡羅
│   └── models/            # 物理模型
│       ├── __init__.py
│       ├── hubbard.py     # Hubbard模型
│       └── bose_hubbard.py # Bose-Hubbard模型
├── examples/              # 範例腳本
│   ├── hubbard_vmc.py
│   ├── bose_hubbard_pimc.py
│   └── phase_transition.py
├── tests/                 # 測試檔案
├── requirements.txt       # 依賴套件
└── README.md

```

## 安裝

```bash
pip install -r requirements.txt
```

## 快速開始

### 範例1: Hubbard模型的變分蒙地卡羅

```python
from qmc.models import HubbardModel
from qmc.methods import VMC
from qmc.nqs import RBMQuantumState

# 建立模型
model = HubbardModel(L=4, U=4.0, t=1.0)

# 建立神經網路量子態
nqs = RBMQuantumState(model.n_sites, hidden_dim=16)

# 執行VMC
vmc = VMC(model, nqs)
results = vmc.run(n_steps=10000, n_walkers=100)
```

### 範例2: Bose-Hubbard模型的相變研究

```python
from qmc.models import BoseHubbardModel
from qmc.methods import PIMC

# 建立模型
model = BoseHubbardModel(L=6, U=1.0, t=1.0, mu=0.5)

# 執行PIMC
pimc = PIMC(model)
results = pimc.run(n_steps=5000, beta=2.0)
```

## 參考文獻

- Carleo, G., & Troyer, M. (2017). Solving the quantum many-body problem with artificial neural networks. Science, 355(6325), 602-606.
- Foulkes, W. M. C., et al. (2001). Quantum Monte Carlo simulations of solids. Reviews of Modern Physics, 73(1), 33.
- Ceperley, D. M. (1995). Path integrals in the theory of condensed helium. Reviews of Modern Physics, 67(2), 279.

## 授權

MIT License

