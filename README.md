# MP219 - College Canteen Lunch Rush (Guwahati)

## Project Details

- Course: SCSE3040 - MLOps
- Group ID: G194
- Problem ID: MP219
- Batch: 32
- Faculty: Saurabh Shrivastava

## Team Members

| Name | Enrollment Number |
|---|---|
| Saakshi Sharma | S24CSEU0436 |
| Rudra Pratap Chauhan Singh | S24CSEU2434 |
| Nishant Bisht Singh | S24CSEU1446 |
| Krish Verma | S24CSEU2353 |

## Project Overview

The objective is to develop an end-to-end MLOps
pipeline to predict the number of lunch plates
sold in a college canteen.

The project uses regression with Mean Absolute
Error (MAE) as the evaluation metric.

## Technologies

- Python
- Pandas and NumPy
- Git and GitHub
- DVC
- MLflow
- Scikit-learn
- FastAPI
- Docker
- Kubernetes
- GitHub Actions

## Dataset

| Dataset | Rows | Purpose |
|---|---:|---|
| train.csv | 1030 | Model training |
| later.csv | 300 | Monitoring and retraining |
| holdout_features.csv | 250 | Final predictions |
| train_cleaned.csv | 1000 | Cleaned training data |

## Data Cleaning

- Removed 30 duplicate rows.
- Corrected B-Block queue duration from seconds
  to minutes.
- Replaced 34 invalid temperature readings with
  missing values.
- Corrected 15 target typos after deduplication.
- Standardized 18 day-type variants into 3 categories.
- No leaking column or stuck sensor was detected.

Missing temperatures will be handled during
model preprocessing.

## Installation

Create a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the original datasets in data/raw/
using the files provided by the faculty.

## Run Data Inspection

```powershell
python src/canteen/data_inspection.py
```

## Run Data Cleaning

```powershell
python src/canteen/data_cleaning.py
```

## Verify DVC

```powershell
python -m dvc status
```

## Project Status

- [x] Environment setup
- [x] DVC dataset tracking
- [x] Data inspection
- [x] Data cleaning
- [x] Data validation
- [ ] Model training
- [ ] MLflow tracking
- [ ] FastAPI deployment
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] CI/CD
- [ ] Drift monitoring and retraining
- [ ] Holdout predictions