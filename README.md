# PIEL Project Documentation

## Overview
The code used in paper ##Physics-Informed Ensemble Learning for Lithium-Ion Battery SOH Estimation with Adaptive Physical-statistical Feature Fusion is inthis project.
## Features
- Adaptive multi-source feature fusion
- Robust light estimation algorithms
- Comprehensive evaluation metrics for model performance
- User-friendly scripts for data processing and model evaluation

## Installation
To install the dependencies and set up the project, please run the following commands:
```bash
pip install -r requirements.txt
```

## Quick Start
To quickly start with the project, you can use the provided scripts as follows:
```bash
python main.py --config config.yaml
```

## Datasets
The datasets used for training and evaluation are available in the `data/` directory. Please ensure you have proper access to the datasets before running the scripts.

## Model Architecture
The model leverages convolutional neural networks (CNNs) combined with feature fusion techniques to enhance performance. A detailed architecture diagram is provided in the `docs/` directory.

## Evaluation Metrics
The evaluation of model performance is based on several metrics including:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²)

## Project Structure
The overall structure of the project is organized as follows:
```
PIEL/
├── data/
├── docs/
├── scripts/
├── main.py
└── requirements.txt
```

## Usage Tips
- Always check for the latest updates in the repository.
- Use virtual environments to manage dependencies effectively.

## Scripts Explanation
- `main.py`: The main script that runs the light estimation model.
- `data_preprocess.py`: Script for preprocessing datasets before training.
- `evaluate.py`: Script for evaluating the model's performance.

## Citations
For academic usage or reference, please cite this project as follows:
```
@misc{piel2026,
    title={PIEL: Image-based Estimation of Light},
    author={Author Name},
    year={2026},
    publisher={GitHub},
    url={https://github.com/moming-z/PIEL-for-LIB-SOH-Estimation-with-Adaptive-Multi-Source-Feature-Fusion},
}
```
