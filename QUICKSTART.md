# 快速入門指南

本指南將幫助您快速開始使用量子蒙地卡羅模擬專案。

## 安裝

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 安裝專案（可選）

```bash
pip install -e .
```

## 快速開始

### 範例 1: Hubbard 模型的變分蒙地卡羅

```python
from qmc.models import HubbardModel
from qmc.methods import VMC
from qmc.nqs import RBMQuantumState

# 建立模型
model = HubbardModel(L=4, U=4.0, t=1.0)

# 建立神經網路量子態
nqs = RBMQuantumState(model.n_sites, hidden_dim=16)

# 執行 VMC
vmc = VMC(model, nqs)
results = vmc.run(n_steps=5000, n_walkers=100)

print(f"基態能量: {results['energy']:.6f} ± {results['energy_error']:.6f}")
```

### 範例 2: Bose-Hubbard 模型的相變研究

```python
from qmc.models import BoseHubbardModel
from qmc.methods import PIMC

# 建立模型
model = BoseHubbardModel(L=6, U=1.0, t=1.0, mu=0.5)

# 執行 PIMC
pimc = PIMC(model)
results = pimc.run(n_steps=5000, beta=2.0)

print(f"能量: {results['energy']:.4f}")
print(f"序參數: {results['order_parameter']:.4f}")
print(f"超流密度: {results['superfluid_density']:.4f}")
```

## 執行範例腳本

### 1. Hubbard 模型 VMC 範例

```bash
python examples/hubbard_vmc.py
```

這個腳本會：
- 建立 4x4 的 Hubbard 模型
- 使用 RBM 量子態
- 執行變分蒙地卡羅模擬
- 優化波函數參數
- 產生結果圖表

### 2. Bose-Hubbard 模型 PIMC 範例

```bash
python examples/bose_hubbard_pimc.py
```

這個腳本會：
- 建立 6x6 的 Bose-Hubbard 模型
- 使用路徑積分蒙地卡羅
- 研究溫度相關的相變
- 研究 U/t 相關的相變
- 產生相變圖表

### 3. 綜合相變分析

```bash
python examples/phase_transition.py
```

這個腳本會：
- 進行系統性的相變研究
- 執行有限尺寸標度分析
- 識別臨界點
- 產生詳細的相變圖表

## 核心概念

### 量子態類型

1. **RBM 量子態** (`RBMQuantumState`)
   - 受限玻爾茲曼機
   - 適合強關聯系統
   - 計算效率高

2. **MLP 量子態** (`MLPQuantumState`)
   - 多層感知器
   - 靈活的架構
   - 適合複雜系統

3. **CNN 量子態** (`CNNQuantumState`)
   - 卷積神經網路
   - 捕捉空間相關性
   - 適合晶格系統

### QMC 方法

1. **變分蒙地卡羅 (VMC)**
   - 結合神經網路量子態
   - 優化變分參數
   - 計算基態能量

2. **輔助場 QMC (AFQMC)**
   - 處理費米子系統
   - 使用 Hubbard-Stratonovich 變換
   - 處理符號問題

3. **路徑積分蒙地卡羅 (PIMC)**
   - 有限溫度計算
   - 玻色子系統
   - 相變研究

### 物理模型

1. **Hubbard 模型**
   - 強關聯電子系統
   - 參數：U (相互作用), t (躍遷), mu (化學勢)

2. **Bose-Hubbard 模型**
   - 超流-絕緣體相變
   - 參數：U (相互作用), t (躍遷), mu (化學勢)

## 參數調整建議

### VMC 參數
- `n_steps`: 1000-10000 (更多步驟 = 更準確)
- `n_walkers`: 50-200 (更多行走者 = 更穩定)
- `n_equil`: 500-2000 (平衡步驟)
- `learning_rate`: 0.001-0.1 (學習率)

### PIMC 參數
- `n_steps`: 2000-10000
- `beta`: 0.5-5.0 (逆溫度，beta = 1/T)
- `n_tau`: 10-50 (時間切片數)

### NQS 參數
- `hidden_dim`: 8-64 (隱藏單元數，越大越靈活但計算更慢)

## 常見問題

### Q: 如何選擇合適的量子態？
A: 
- 小系統 (< 20 sites): RBM
- 大系統或需要空間相關性: CNN
- 複雜系統: MLP

### Q: VMC 和 PIMC 的區別？
A:
- VMC: 零溫基態，需要變分波函數
- PIMC: 有限溫度，不需要變分波函數

### Q: 如何判斷結果是否收斂？
A:
- 檢查能量歷史是否穩定
- 檢查能量方差是否足夠小
- 增加模擬步數並比較結果

## 進階使用

### 自定義模型

```python
from qmc.core.base import QuantumModel, QuantumState

class MyModel(QuantumModel):
    def hamiltonian(self, config):
        # 實現您的哈密頓量
        pass
    
    def local_energy(self, config, state):
        # 實現局部能量
        pass
```

### 自定義量子態

```python
from qmc.core.base import QuantumState

class MyQuantumState(QuantumState):
    def amplitude(self, config):
        # 實現波函數振幅
        pass
```

## 參考文獻

- Carleo & Troyer (2017): Neural Network Quantum States
- Foulkes et al. (2001): Quantum Monte Carlo simulations
- Ceperley (1995): Path integrals in condensed matter

## 取得幫助

如有問題，請查看：
1. README.md - 專案概述
2. 範例腳本 - 實際使用案例
3. 原始碼註解 - 詳細實現說明

