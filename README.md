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
IOM_soh_try.py
```
comparative experiment:
```bash
IOM_soh_duibi_cnn_mlp.py
IOM_soh_duibi_patchtst_informer.py
```

ablation experiment:
```bash
IOM_soh_xiaorong.py
```

transfer learning and few-shot experiment:
```bash
IOM_soh_fine_tune.py
IOM_soh_few_shot.py
```
## Model
The ‘model’ folder contains all the model structures used in this paper

### Dataset Storage / Data Structure

* **TJU Datasets:**
    * **Physical information & Statistical features:** Stored in the `NCA` and `NCM` folders.
    * **Labels & Capacity data:** Stored in the `TJU` folder/file.

* **Oxford Datasets:**
    * **Physical information:** Stored in the `1R_IOM_ga_data_250630` folder.
    * **Statistical features:** Stored in the `oxford_statisitc_data_20250701` folder.
    * **Labels:** Stored in the `oxford_soh_20251026` folder.
    * **Capacity data:** Saved in the `oxford` folder.

## Datasets
TJU datasets 的物理信息和统计特征保存在NCA和NCM文件夹中，标签和容量数据保存在TJU文件中
Oxford datasets的物理信息存储在1R_IOM_ga_data_250630文件夹中，统计特征存储在oxford_statisitc_data_20250701文件夹中，标签存储在oxford_soh_20251026文件夹中，容量数据保存在oxford文件夹中



## Evaluation Metrics
The evaluation of model performance is based on several metrics including:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²)
  


