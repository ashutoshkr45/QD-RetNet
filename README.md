# Quantized Distillation Framework for Retinal Disease Classification

A novel framework implementing DML, knowledge distillation between Computational Fundus Photography (CFP) and Optical Coherence Tomography (OCT) images for retinal disease classification, enhanced with quantization-aware training for efficient deployment.

## 📊 Dataset: Topcon-MM

* **Type**: Multiclass & Multilabel Classification
* **Classes**: 11 (Normal, dAMD, wAMD, DR, CSC, PED, MEM, FLD, EXU, CNV, RVO)
* **Fundus Images**: 1520 (Train: 1150, Test: 370)
* **OCT Images**: 1435 (Train: 1084, Test: 351)

## 📈 Results

| Metrics | Single-Modal (Fundus) | Single-Modal (OCT) | FDDM (I) ❌ | FDDM + QAT (I) ✓ | ODDM (II) ❌ | ODDM + QAT (II) ✓ | DDM + DML + QAT (I, II) ✓ |
|---------|----------------------|-------------------|-------------|-----------------|--------------|------------------|--------------------------|
| Filename  | `train_fundus.py` | `train_oct.py` | `train_fddm.py` | `train_I.py` | `train_oddm.py` | `train_II.py` | `main.py` |
| MAP | 50.56(±3) | 66.44(±4) | 69.06(±3) | 68.6 | 56.53 | 54.27 | 68.9, 50.93 |
| Accuracy | 83.19(±3) | 87.73(±2) | 90.27(±2) | 87.32 | 83.83 | 84.46 | 87.95, 82.98 |
| AUC | 79.97(±3) | 87.73(±2) | 89.06(±1) | 88.38 | 83.44 | 80.83 | 88.69, 77.33 |
| F1 Score | 54.95(±7) | 64.16(±6) | 69.17(±6) | 68.66 | 59.01 | 54.27 | 68.63, 53.66 |
| Size (MB) | 97.8 | 97.8 | 97.8 | 24.9 | 97.8 | 24.9 | 24.9, 24.9 |

**Note:**
- I: Fundus Teacher → OCT Student
- II: OCT Teacher → Fundus Student
- ✓: Quantized Model
- ❌: Not Quantized

## 📁 Data Folder Structure

```
image_data/
└── topcon-mm/
    ├── train/
    │   ├── cfp.txt
    │   ├── oct.txt
    │   └── Images/
    │       ├── fundus-images/
    │       └── oct-images/
    ├── val/
    │   ├── cfp.txt
    │   └── oct.txt
    └── test/
        ├── cfp.txt
        └── oct.txt
```

## 🛠️ Dependencies

* Python 3.7.10

Other dependencies are listed in `requirements.txt`

## 🚀 Getting Started

Code is available at:
```
/raid/home/dgx1575/Ashutosh/quant_fddm
```

## 📚 References

### Knowledge Distillation
1. Wang, L., Dai, W., Jin, M., Ou, C., & Li, X. (2023). [Fundus-Enhanced Disease-Aware Distillation Model for Retinal Disease Classification from OCT Images](https://arxiv.org/pdf/2308.00291)
2. [Official Implementation Repository: FDDM](https://github.com/xmed-lab/FDDM)

### Quantization Aware Training
1. [PyTorch Documentation: Quantization](https://pytorch.org/docs/stable/quantization.html)
2. [PyTorch Tutorial: Quantization Aware Training](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

