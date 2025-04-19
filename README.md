# Re-ProtGNN

A re-implementation of the AAAI 2022 paper:  
**ProtGNN: Towards Self-Explaining Graph Neural Networks**  
[arXiv:2112.00911](https://arxiv.org/abs/2112.00911)

> **Developer:** Yuanqi Xue  
> **Project Start Date:** January 14, 2025

---

## Project Overview

Re-ProtGNN aims to replicate and study the paper *ProtGNN: Towards Self-Explaining Graph Neural Networks* by Zhang et al. This project includes source code, documentation, and unit/system tests.

---

## Directory Structure

```
├── .github/
│ └── workflows/ci.yml # GitHub Actions file for Continuous Integration (CI) 
├── data/ # Input datasets (MUTAG, etc.)
├── docs/ # Documentation: SRS, MG, MIS, VnV Plan
├── refs/ # Reference materials and citations
├── results/ # Plots and logs
├── src/ # Source code
│ ├── data_processing/ # dataUtils.py
│ ├── evaluaton/ # inference.py, explanation.py
│ ├── models/ # GNN model components, train.py
│ ├── utils/ # Logging, visualization, config parsers: outputUtils.py, Configures.py
│ └── main.py # Entry point for training and inference
├── test/ # System and unit tests
├── requirements.txt # Python dependencies
```

---


## Included Results

A Jupyter notebook containing pre-run results is available in the `notebooks` directory.


---

## Requirements

- Python 3.8+
- PyTorch 1.8.1
- PyTorch Geometric 2.0.2

Install all dependencies using:

```bash
pip install -r requirements.txt
```


---





## Usage

1. **Set hyperparameters**  
   Edit `./utils/Configures.py` to customize model and training settings.

2. **Run the model**

```bash
python src/main.py
```

> This implementation has been tested and verified on the **MUTAG** datasets.

## Reference

Original Paper:

```
@article{zhang2021protgnn,
  title={ProtGNN: Towards Self-Explaining Graph Neural Networks},
  author={Zhang, Zaixi and Liu, Qi and Wang, Hao and Lu, Chengqiang and Lee, Cheekong},
  journal={arXiv preprint arXiv:2112.00911},
  year={2021}
}
```

