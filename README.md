# A Unified Framework for Tabular Generative Modeling: Loss Functions, Benchmarks, and Improved Multi-objective Bayesian Optimization Approaches

[![TMLR](https://img.shields.io/badge/TMLR-2026-blue)](https://openreview.net/forum?id=RPZ0EW0lz0)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-green)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13.1-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper published in *Transactions on Machine Learning Research* (2026).

> **Minh H. Vu, Daniel Edler, Carl Wibom, Tommy Löfstedt, Beatrice Melin, Martin Rosvall**
> Department of Diagnostics and Intervention / Physics / Computing Science, Umeå University
> [OpenReview](https://openreview.net/forum?id=RPZ0EW0lz0)

---

## Abstract

Deep generative models (DGMs) offer a promising solution to data scarcity by synthesizing realistic tabular data. However, current approaches often fail to preserve feature correlations and distributions during training, struggle with multi-metric hyperparameter selection, and lack comprehensive evaluation protocols. This work addresses these shortcomings with a unified framework that integrates training, hyperparameter tuning, and evaluation.

We introduce **(1)** a novel **correlation- and distribution-aware loss function** that regularizes DGMs to faithfully reproduce the statistical structure of real tabular data, with theoretical stability and consistency guarantees; **(2)** **Iterative Objective Refinement Bayesian Optimization (IORBO)**, a rank-based multi-objective aggregation strategy that resolves the unit-heterogeneity problem inherent in standard Bayesian optimization (SBO); and **(3)** a **comprehensive open-source benchmarking framework** spanning twenty real-world datasets and ten established tabular DGM baselines, evaluated across statistical similarity, Train-Synthetic-Test-Real (TSTR), and ML augmentation tasks.

The proposed loss function achieves win rates of **0.611** (TSTR) and **0.551** (augmentation) over vanilla loss functions, while IORBO significantly outperforms SBO-Mean (win rate **0.591**) and SBO-Median (win rate **0.561**).

---

## Key Features

- **Correlation-Aware Loss** (`engine/custom_loss.py`): A differentiable pairwise feature-correlation regularizer that penalizes discrepancies between the off-diagonal correlation matrices of real and synthetic data batches. Proven numerically stable via sub-Gaussian gradient bounds (Proposition 4.2).

- **Distribution-Aware Loss** (`engine/custom_loss.py`): A moment-matching regularizer that aligns up to *H* statistical moments (mean, variance, and higher-order standardized moments) between real and synthetic features. Proven numerically consistent under high-probability moment-matching (Proposition 4.4, Remark 4.5).

- **Unified DGM Loss Integration**: The proposed regularizers augment the native objectives of GAN (Eq. 9), VAE (Eq. 10), and DDPM (Eq. 11) architectures as auxiliary terms weighted by hyperparameters α and β, preserving each model's primary optimization criterion.

- **IORBO — Iterative Objective Refinement Bayesian Optimization** (`engine/utils/hyperopt_utils.py`): A surrogate-model-based optimization strategy that dynamically re-ranks all historical evaluations at each iteration using cross-sample rank averaging, resolving bias from heterogeneous metric scales without incurring additional asymptotic computational cost (O(mHB) per iteration, matching SBO).

- **Ten DGM Baselines** (`models/`): CTGAN, TVAE, CopulaGAN, DP-CGANS, CTAB-GAN, and TabDDPM (MLP/ResNet backbones), each with optional conditional vector support and the proposed loss function.

- **Multi-task Evaluation Suite** (`engine/evaluate_technical_paper.py`): Three evaluation axes — (1) statistical similarity (KL divergence, Pearson Chi-Square, Kolmogorov–Smirnov, Dimension-wise Probability); (2) ML TSTR performance (SVM, Random Forest, Bagging, XGBoost across balanced accuracy, F-score, AUC, MAE, MSE, R²); and (3) **ML augmentation** — the first systematic benchmark task assessing whether synthetic data improves models trained on combined real+synthetic sets.

- **Rigorous Statistical Testing** (`scripts/perform_friedman_nemenyi_test.py`): Friedman test followed by Nemenyi post-hoc pairwise comparisons, reported with a six-tier significance notation (Table 3 in the paper).

- **Twenty Benchmark Datasets** (`engine/dataset_helper/`): Covering regression, binary classification, and multiclass tasks, ranging from 768 rows (Diabetes-ML) to 277,640 rows (Credit), with up to 144 features (MNIST12).

---

## Repository Structure

```text
TabGen-Framework/
│
├── engine/                              # Core framework engine
│   ├── custom_loss.py                   # §3.1 — Correlation & distribution-aware losses
│   ├── utils/
│   │   ├── hyperopt_utils.py            # §3.2 — IORBO & SBO implementations
│   │   ├── eval_utils.py                # §3.3 — Evaluation metric utilities
│   │   ├── train_utils.py               # §5.2 — Training helpers
│   │   ├── data_utils.py                # §5.1 — Data preprocessing utilities
│   │   ├── model_utils.py               # Model initialization utilities
│   │   ├── path_utils.py                # Experiment path management
│   │   └── io_utils.py                  # I/O and argument conversion utilities
│   ├── evaluate_technical_paper.py      # §3.3 — Statistical, TSTR & augmentation evaluation
│   ├── experiment_technical_paper.py    # §3.6 — Benchmarking pipeline orchestration
│   ├── dataset_helper/
│   │   ├── public.py                    # §5.1 — All 20 dataset class definitions
│   │   ├── base.py                      # Dataset base class & registry
│   │   └── preprocessing.py            # Data encoding & preprocessing
│   ├── datasets.py                      # Dataset factory (get_dataset)
│   ├── config.py                        # Global configuration & dataset metadata
│   ├── logger.py                        # Experiment logging
│   ├── ctgan_data_sampler.py            # CTGAN conditional sampler
│   ├── ctgan_data_transformer.py        # CTGAN data transformer
│   ├── dpcgans_data_sampler.py          # DP-CGANS conditional sampler
│   ├── dpcgans_data_transformer.py      # DP-CGANS data transformer
│   ├── rdp_accountant.py                # Rényi differential privacy accounting
│   └── analysis.py                      # Post-hoc analysis utilities
│
├── models/                              # §3.6 — DGM implementations
│   ├── ctgan.py                         # CTGAN (Xu et al., 2019)
│   ├── tvae.py                          # TVAE (Xu et al., 2019)
│   ├── copulagan.py                     # CopulaGAN
│   ├── dpcgans.py                       # DP-CGANS (Sun et al., 2023)
│   ├── autoencoder.py                   # Shared autoencoder components
│   ├── base.py                          # Abstract model base class
│   ├── CTAB/                            # CTAB-GAN (Zhao et al., 2021)
│   │   ├── ctabgan.py
│   │   ├── synthesizer/ctabgan_synthesizer.py
│   │   └── pipeline/data_preparation.py
│   └── tab_ddpm/                        # TabDDPM (Kotelnikov et al., 2023)
│       ├── tab_ddpm/
│       │   ├── gaussian_multinomial_diffusion.py
│       │   └── modules.py
│       └── scripts/
│           ├── train.py
│           ├── sample.py
│           └── tune_ddpm.py
│
├── scripts/                             # Experiment entry points
│   ├── main_technical_paper.py          # §5.2 — DGM training script
│   ├── main_optimize_technical_paper.py # §3.4 — Bayesian optimization loop (IORBO/SBO)
│   ├── perform_friedman_nemenyi_test.py # §3.5 — Friedman–Nemenyi statistical tests
│   ├── envs/setup.sh                    # Environment installation script
│   └── apptainer/biobank.def           # HPC container definition (Apptainer/Singularity)
```

---

## Environment Setup

The framework was developed and evaluated on NVIDIA A100 GPUs (40 GB VRAM) with Intel Xeon Gold 6338 CPUs (256 GB DDR4 RAM) running CUDA 11.6. An [Apptainer](https://apptainer.org/) container definition is provided for HPC environments (`scripts/apptainer/biobank.def`).

### Step 1 — Install Anaconda

```bash
wget "https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh"
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b
```

### Step 2 — Create the Conda Environment

This step requires the `libmamba` solver and the RAPIDS AI channel for GPU-accelerated ML evaluation (`cuML`).

```bash
conda create --solver=libmamba -y -n tabgen \
    -c rapidsai -c conda-forge -c nvidia \
    cudf=23.10 cuml=23.10 python=3.9 cuda-version=11.2 \
    jupyterlab dash
conda activate tabgen
```

### Step 3 — Install Python Dependencies

```bash
# Tabular generative modeling & SDV
pip install ctgan sdv==1.8.0

# Differential privacy
pip install opacus

# Bayesian optimization
pip install hyperopt

# Downstream ML evaluation
pip install xgboost==1.7.6 scikit_posthocs researchpy

# Utilities
pip install scipy matplotlib seaborn termcolor tabulate autoflake rich wandb

# PyTorch (CUDA 11.6)
pip uninstall -y torch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

> **Note:** Additional packages (`imgui`, `monai`, `nilearn`, `SimpleITK`, `comet-ml`) are listed in `scripts/envs/setup.sh` and may be required for specific sub-modules or HPC workflows.

### Step 4 — Configure the Python Path

```bash
conda activate tabgen
export PYTHONPATH=${PWD}:$PYTHONPATH
```

---

## Usage and Reproducibility

All experiments in the paper follow a two-stage workflow: **(1) DGM training** with a specified loss function, and **(2) Bayesian optimization** of hyperparameters using IORBO or SBO.

### Stage 1: Train a DGM

Use `scripts/main_technical_paper.py` to train a single DGM configuration.

**General syntax:**

```bash
python -W ignore scripts/main_technical_paper.py \
    --arch <MODEL> \
    --dataset <DATASET> \
    --loss_version <LOSS> \
    --is_test 0
```

**Key arguments:**

| Argument              | Description                             | Choices / Default                                                                                |
|-----------------------|-----------------------------------------|--------------------------------------------------------------------------------------------------|
| `--arch`              | DGM architecture                        | `ctgan`, `tvae`, `copulagan`, `dpcgans`, `ctab`                                                  |
| `--dataset`           | Dataset name                            | e.g., `adult`, `abalone`, `credit`, `mnist12`                                                    |
| `--loss_version`      | Loss function variant                   | `0` (vanilla), `2` (correlation + distribution), `4` (correlation only), `5` (distribution only) |
| `--is_condvec`        | Enable conditional vector               | `1` (enabled), `0` (disabled)                                                                    |
| `--is_loss_corr`      | Weight α for correlation loss           | float, default `1.0`                                                                             |
| `--is_loss_dwp`       | Weight β for distribution loss          | float, default `1.0`                                                                             |
| `--n_moment_loss_dwp` | Number of moments H                     | `1`–`4`, default `4`                                                                             |
| `--epochs`            | Training epochs                         | default `10000`                                                                                  |
| `--batch_size`        | Batch size                              | default `16000`                                                                                  |
| `--is_test`           | Quick test mode (2000 rows, 100 epochs) | `1` (test), `0` (full)                                                                           |

**Example — Reproduce the vanilla CTGAN baseline on Adult:**

```bash
python -W ignore scripts/main_technical_paper.py \
    --arch ctgan \
    --dataset adult \
    --loss_version 0 \
    --epochs 3000 \
    --batch_size 16000 \
    --is_condvec 1 \
    --is_test 0
```

**Example — Train CTGAN with the proposed correlation + distribution loss (loss version 2):**

```bash
python -W ignore scripts/main_technical_paper.py \
    --arch ctgan \
    --dataset adult \
    --loss_version 2 \
    --is_loss_corr 1.0 \
    --is_loss_dwp 1.0 \
    --n_moment_loss_dwp 4 \
    --is_condvec 1 \
    --is_test 0
```

**Example — Train TVAE with the proposed loss on the Credit dataset:**

```bash
python -W ignore scripts/main_technical_paper.py \
    --arch tvae \
    --dataset credit \
    --loss_version 2 \
    --is_condvec 1 \
    --is_test 0
```

---

### Stage 2: Hyperparameter Optimization (IORBO / SBO)

Use `scripts/main_optimize_technical_paper.py` to run the full Bayesian optimization loop. This script orchestrates repeated calls to Stage 1, evaluates each trial across all metric categories, and updates the surrogate model via IORBO or SBO.

**Example — IORBO with the proposed loss on CTGAN / Adult:**

```bash
python -W ignore scripts/main_optimize_technical_paper.py \
    --arch ctgan \
    --dataset adult \
    --loss_version 2 \
    --bo_method ior \
    --max_trials 30 \
    --module public \
    --is_test 0
```

**Example — Standard Bayesian Optimization (mean aggregation) for comparison:**

```bash
python -W ignore scripts/main_optimize_technical_paper.py \
    --arch ctgan \
    --dataset adult \
    --loss_version 0 \
    --bo_method sbo \
    --bo_method_agg mean \
    --max_trials 30 \
    --is_test 0
```

**Key optimization arguments:**

| Argument          | Description               | Choices / Default                        |
|-------------------|---------------------------|------------------------------------------|
| `--bo_method`     | Optimization strategy     | `ior` (IORBO), `sbo` (Standard BO)       |
| `--bo_method_agg` | Aggregation for SBO       | `mean`, `median`                         |
| `--max_trials`    | Number of BO iterations   | default `30`                             |
| `--module`        | Evaluation scope          | `public` (open data), `gm`, `dp`, `gmdp` |
| `--is_condvec`    | Enable conditional vector | `1` (enabled), `0` (disabled)            |

Optimization results and trial logs are saved under `database/optimization/` (IORBO) or `database/optimization_sbo_<agg>/` (SBO).

---

### Stage 3: Statistical Testing

After collecting results across all models and datasets, run the Friedman–Nemenyi post-hoc tests (§3.5) to assess statistical significance:

```bash
python scripts/perform_friedman_nemenyi_test.py
```

This reproduces the significance tables (Tables 5–8 in the paper) using the notation defined in Table 3 (`++`, `+`, `0`, `−`, `−−`).

---

### Quick Smoke Test

To verify the installation end-to-end using reduced data and a single BO trial:

```bash
python -W ignore scripts/main_optimize_technical_paper.py \
    --arch ctgan \
    --dataset adult \
    --loss_version 2 \
    --bo_method ior \
    --max_trials 1 \
    --is_test 1
```

---

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@article{vu2026a,
  title   = {A Unified Framework for Tabular Generative Modeling: Loss Functions,
             Benchmarks, and Improved Multi-objective Bayesian Optimization Approaches},
  author  = {Minh Hoang Vu and Daniel Edler and Carl Wibom and Tommy L{\"o}fstedt
             and Beatrice Melin and Martin Rosvall},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=RPZ0EW0lz0},
}
```

---

## Acknowledgements

The computations and data handling were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) and the Swedish National Infrastructure for Computing (SNIC) at the Uppsala Multidisciplinary Center for Advanced Computational Science (UPPMAX), partially funded by the Swedish Research Council under grant agreements no. 2022-06725 and no. 2018-05973. This work was further supported by the WASP–DDLS postdoctoral grant *Visualization and de-Identification of Biobank Data to Propel Precision Medicine Research* (KAW 2023-03705), by the Swedish Cancer Foundation (24 3406 Pj 01 H; 21 1384 Pj 01 H), and by Umeå University Infrastructure funding (FS 2.1.6-1689-24).
