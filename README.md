# MLOps Assignment 3 – End-to-End ML Pipeline (California Housing)

This project implements a full MLOps pipeline using the **California Housing dataset**. It demonstrates model training, containerization, CI/CD automation, and manual quantization.

---

## Public Links

- **GitHub Repository:** [mlops-pytorch](https://github.com/Ashwini-Gerigenti/mlops-pytorch)
- **DockerHub Image:** [ashwinigerigenti/california-model](https://hub.docker.com/repository/docker/ashwinigerigenti/california-model)

---

## Branch Structure

| Branch         | Purpose                                           |
|----------------|---------------------------------------------------|
| `main`         | Initial setup – README, .gitignore               |
| `dev`          | Model training using scikit-learn (train.py)     |
| `docker_ci`    | Dockerfile, predict.py, GitHub Actions CI/CD     |
| `quantization` | Manual quantization and PyTorch inference        |

---

## Dataset

- **Source**: `sklearn.datasets.fetch_california_housing`
- **Target**: Median house value (continuous regression task)
- **Features**: 8 numerical predictors

---

## Pipeline Stages

### Step 1: `main` Branch
- Initialized with `.gitignore` and `README.md`.

### Step 2: `dev` Branch
- Trains a **Linear Regression** model using `scikit-learn`.
- Saves model to `model.joblib`.

### Step 3: `docker_ci` Branch
- Adds `Dockerfile`, `predict.py`, and `requirements.txt`.
- GitHub Actions CI:
  - Trains model
  - Builds Docker image
  - Tests model inside container
  - Pushes image to DockerHub

### Step 4: `quantization` Branch
- Manually quantizes `model.joblib` weights to `int8`.
- Reconstructs and runs inference using a **PyTorch model**.
- Saves:
  - `unquant_params.joblib`
  - `quant_params.joblib`

---

## Results

| Metric         | Scikit-Learn Model | Quantized PyTorch Model |
|----------------|--------------------|-------------------------|
| **R² Score**   | 0.57               | 0.556                   |
| **Model Size** | 408 Bytes          | 476 Bytes               |

> Note: Quantized model performance is close to original, with smaller weights and acceptable error margin.

---

## How to Run

### Training (scikit-learn)
```bash
python train.py
