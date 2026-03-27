# PIEL Project Documentation

## Overview
This repository contains the official code for the paper "Physics-Informed Ensemble Learning for Lithium-Ion Battery SOH Estimation with Adaptive Physical-Statistical Feature Fusion".
## Installation
To install the dependencies and set up the project, please run the following commands:
```bash
pip install torch=2.4.1
```

## script
conventional experiment:
```bash
python IOM_soh_try.py
```
comparative experiment:
```bash
python IOM_soh_duibi_cnn_mlp.py
python IOM_soh_duibi_patchtst_informer.py
```

ablation experiment:
```bash
python IOM_soh_xiaorong.py
```

transfer learning and few-shot experiment:
```bash
python IOM_soh_fine_tune.py
python IOM_soh_few_shot.py
```
## Model
The ‘model’ folder contains all the model structures used in this paper


## 📂 Project Structure

The repository is organized as follows:

```text
├── 1R_IOM_ga_data_250630/           # Physical information for the Oxford dataset
├── NCA/                             # Physical & statistical data for TJU dataset (NCA cells, includes examples)
├── NCM/                             # Physical & statistical data for TJU dataset (NCM cells, includes examples)
├── TJU/                             # Labels & capacity data for the TJU dataset
├── model/                           # Main model architectures and network structures
├── oxford/                          # Capacity data for the Oxford dataset
├── oxford_soh_20251026/             # Labels for the Oxford dataset
├── oxford_statisitc_data_20250701/  # Statistical features for the Oxford dataset
├── results/                         # Directory for saving outputs, logs, and experimental results
│   ├── ablation_experiment/         # Results for ablation studies
│   ├── compare_experiment/          # Results for comparative experiments
│   ├── qianyi_experiment/           # Results for transfer learning
│   └── qianyi_experiment2/          
├── IOM_soh_try.py                   # Main program for the primary SOH estimation experiment
├── IOM_soh_xiaorong.py              # Script for ablation experiments
├── IOM_soh_duibi_cnn_mlp.py         # Comparative experiment script (vs. CNN & MLP)
├── IOM_soh_duibi_patchtst_informer.py # Comparative experiment script (vs. PatchTST & Informer)
├── IOM_soh_fine_tune.py             # Script for transfer learning (fine-tuning) experiments
├── IOM_soh_few_shot.py              # Script for few-shot learning experiments
├── dataset_*.py                     # Dataloader scripts for processing TJU and Oxford datasets
└── README.md                        # Project documentation


  


