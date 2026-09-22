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