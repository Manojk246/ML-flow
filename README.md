
# MLflow Project: Garments Worker Productivity

## Setup

```
pip install -r requirements.txt
```

## Train and Track with MLflow

```
python train.py --data-path garments_worker_productivity.csv --experiment-name garments-prod
```

Artifacts (plots, models) and metrics will be logged to the local `mlruns/` directory by default. Open the UI:

```
mlflow ui
```

Then visit http://127.0.0.1:5000 in your browser.
