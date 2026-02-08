# Learning Traffic Flows - Replication Project (Clean Version)

## 📋 Project Overview

This is a **clean replication** of the "Learning Traffic Flows" paper, containing only the essential core files.

- **Created**: 1770546611.2713897
- **Source Project**: GraphGPS-main
- **Goal**: Predict traffic flows using GNNs

---

## 📁 Project Structure

```
GraphGPS-Traffic-Replication-CLEAN/
│
├── create_sioux_data/              # Data generation module
│   ├── load_sioux.py               # Load Sioux Falls network
│   ├── generate_scenarios.py       # LHS sampling for scenarios
│   ├── sue_solver.py               # Frank-Wolfe SUE solver
│   ├── build_pyg_data.py           # Build PyG data objects
│   ├── main_create_dataset.py      # Main data generation entry
│   ├── utils.py                    # Utility functions
│   └── processed_data/             # Generated data storage
│
├── sioux_data/                     # Raw network data
│   └── SiouxFalls_net.tntp         # Sioux Falls network topology
│
├── graphgps/                       # GraphGPS framework
│   ├── encoder/                    # Feature encoders
│   │   └── linear_edge_encoder.py  # [Modified] Edge feature encoder
│   ├── head/                       # Output heads
│   │   └── edge_regression.py      # [New] Edge regression head
│   ├── loader/                     # Data loaders
│   │   ├── master_loader.py        # [Modified] Main loader
│   │   └── dataset/
│   │       └── sioux_falls_traffic.py  # [New] Sioux Falls dataset
│   ├── layer/                      # GNN layers (GatedGCN)
│   ├── network/                    # Network models
│   ├── loss/                       # Loss functions
│   ├── train/                      # Training loops
│   └── config/                     # Configuration system
│
├── configs/GatedGCN/               # Experiment configurations
│   ├── sioux-falls-GatedGCN.yaml                   # Experiment A
│   └── sioux-falls-GatedGCN-with-edge-feats.yaml   # Experiment B
│
├── main.py                         # Training entry point
├── setup.py                        # Installation script
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n graphgps python=3.10
conda activate graphgps

# Install dependencies
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse
pip install numpy scipy tqdm pyyaml networkx
pip install matplotlib seaborn scikit-learn

# Install GraphGPS
pip install -e .
```

### 2. Generate Dataset

```bash
cd create_sioux_data
python main_create_dataset.py --num_samples 10000
```

**Parameters**:
- `--num_samples`: Number of samples to generate (default: 10000)
- `--train_ratio`: Training set ratio (default: 0.6)
- `--val_ratio`: Validation set ratio (default: 0.2)
- `--test_ratio`: Test set ratio (default: 0.2)

### 3. Train Models

#### Experiment A (Baseline)

```bash
cd ..
python main.py --cfg configs/GatedGCN/sioux-falls-GatedGCN.yaml
```

#### Experiment B (With Edge Features)

```bash
python main.py --cfg configs/GatedGCN/sioux-falls-GatedGCN-with-edge-feats.yaml
```

**Expected Time**: ~2.5 hours per experiment (200 epochs)

### 4. Visualize Results (Optional)

```bash
python visualize_results.py
```

---

## 📊 Core Files

### ✅ New Files (9 files)

Files created specifically for Learning Traffic Flows replication:

1. `create_sioux_data/generate_scenarios.py` - LHS sampling
2. `create_sioux_data/build_pyg_data.py` - PyG data construction
3. `create_sioux_data/main_create_dataset.py` - Main data generation
4. `create_sioux_data/utils.py` - Utility functions
5. `graphgps/loader/dataset/sioux_falls_traffic.py` - Custom dataset
6. `graphgps/head/edge_regression.py` - Edge regression head
7. `configs/GatedGCN/sioux-falls-GatedGCN.yaml` - Experiment A config
8. `configs/GatedGCN/sioux-falls-GatedGCN-with-edge-feats.yaml` - Experiment B config
9. `visualize_results.py` - Visualization script

### ✏️ Modified Files (6 files)

Files modified from GraphGPS base:

1. `create_sioux_data/load_sioux.py` - Adapted for Sioux Falls
2. `create_sioux_data/sue_solver.py` - Frank-Wolfe solver
3. `graphgps/loader/dataset/__init__.py` - Register dataset
4. `graphgps/loader/master_loader.py` - Register loader
5. `graphgps/encoder/linear_edge_encoder.py` - Support 3D edge features
6. `graphgps/head/__init__.py` - Register output head

---

## 🎯 Experiment Configuration

### Dataset
- **Samples**: 10,000
- **Split**: Train 60% / Val 20% / Test 20%
- **Nodes**: 24 (11 centroids)
- **Edges**: 76

### Model Architecture
- **GNN Type**: GatedGCN
- **Layers**: 5 MPNN layers
- **Hidden Dimension**: 128
- **Decoder**: concat(h_i, h_j, e_ij)

### Training Parameters
- **Epochs**: 200
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: AdamW
- **Loss**: L1 (MAE)

---

## 📈 Expected Results

Based on the paper, expected metrics:

- **MAE**: < 100 vehicles/hour
- **R²**: > 0.95
- **Spearman**: > 0.98

---

## 📂 Comparison with Full Project

| Project | Files | Description |
|---------|-------|-------------|
| Full GraphGPS-main | 300+ | Includes all examples, tests, other datasets |
| This Clean Version | ~50 | Only core files needed for replication |

**Advantages**:
- ✅ Clear project structure
- ✅ Faster installation and understanding
- ✅ Focus only on Sioux Falls traffic prediction
- ✅ Easy to share and deploy

---

## 🔧 Troubleshooting

### Issue 1: Data Generation Failed

```bash
# Check network file
ls sioux_data/SiouxFalls_net.tntp

# Check Python environment
python -c "import numpy, networkx, torch; print('OK')"
```

### Issue 2: Training Cannot Find Data

```bash
# Verify data is generated
ls create_sioux_data/processed_data/processed/
# Should see: train_dataset.pt, val_dataset.pt, test_dataset.pt
```

### Issue 3: Import Errors

```bash
# Reinstall GraphGPS
pip install -e . --force-reinstall
```

---

## 📚 References

1. **Learning Traffic Flows** - Original paper
2. **GraphGPS** - Base framework
3. **Sioux Falls Network** - Traffic network data

---

## 📧 Contact

For more information, refer to the complete documentation:
- `REPLICATION_FILES_CHECKLIST.md` - File checklist
- `MODIFICATIONS_SUMMARY.md` - Modification summary
- `DATASET_CREATION_GUIDE.md` - Data generation guide

**Happy Replicating!** 🎉
