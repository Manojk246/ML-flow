import argparse
import os
from typing import Dict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_csv
from src.preprocess import preprocess
from src.models import get_model_registry
from src.metrics_plots import regression_metrics, plot_predictions, plot_feature_importance, plot_model_comparison


def run_experiment(data_path: str, experiment_name: str = "default", test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    df = load_csv(data_path)
    X, y, df_proc, log_cols = preprocess(df, target_col='actual_productivity')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = get_model_registry()
    results = []

    mlflow.set_experiment(experiment_name)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "rows": len(df), "test_size": test_size, "random_state": random_state})
            if len(log_cols) > 0:
                mlflow.log_param("log_transformed", ",".join(log_cols))

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = regression_metrics(y_test, y_pred)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(model, name="model")

            fig_pred = plot_predictions(y_test, y_pred, title=f"{name}: Predicted vs Actual")
            pred_path = f"pred_{name}.png"
            fig_pred.savefig(pred_path, dpi=120, bbox_inches='tight')
            mlflow.log_artifact(pred_path, artifact_path="plots")
            os.remove(pred_path)

            if hasattr(model, "feature_importances_"):
                fig_imp = plot_feature_importance(model.feature_importances_, list(X.columns), title=f"{name}: Feature Importance")
                imp_path = f"featimp_{name}.png"
                fig_imp.savefig(imp_path, dpi=120, bbox_inches='tight')
                mlflow.log_artifact(imp_path, artifact_path="plots")
                os.remove(imp_path)

            results.append({"model": name, **metrics})

    results_df = pd.DataFrame(results)

    with mlflow.start_run(run_name="comparison"):
        fig_cmp = plot_model_comparison(results_df, metric='r2')
        cmp_path = "model_comparison.png"
        fig_cmp.savefig(cmp_path, dpi=120, bbox_inches='tight')
        mlflow.log_artifact(cmp_path, artifact_path="plots")
        os.remove(cmp_path)
        mlflow.log_table(results_df, artifact_file="results.json")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="garments_worker_productivity.csv")
    parser.add_argument("--experiment-name", type=str, default="garments-prod")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
