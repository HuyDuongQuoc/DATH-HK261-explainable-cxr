# Explainable AI in Lung Anomalies detection with Location-aware Loss.
(The title is temporarily not decided).

## Setting up the environment
- Make sure to have kaggle account.
- Using python.

- First activate the virtual environment of python:
```sh
python -m venv venv
source venv/bin/activate
```
- Download required packages:
```sh
pip install -r requirements.txt
```
- Download the Kaggle resized dataset version. Note: Make sure to pass in the Kaggle Username and the Kaggle API Token at .env before running. You can checkout .env.example and scripts/prepare_data.py for a clear format acknowledgement.
```sh
python scripts/prepare_data.py
```

## EDA

## Data protocol 85/15 + external test

Protocol hiện hành là `vindr_s8515_exttest_v1`: 12.750 train, 2.250 validation cố định và 3.000 external-test images có ground truth. Xem [docs/data_protocol_s8515.md](docs/data_protocol_s8515.md) để chạy đúng workflow trên codebase hiện tại.
