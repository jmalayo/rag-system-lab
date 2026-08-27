# log_metrics, tracked_run

from __future__ import annotations

from contextlib import contextmanager

import mlflow

from shared.settings import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

def _sanitize(name: str) -> str:
    return name.replace("@", "_at_")

def log_metrics(metrics: dict) -> None:

    numeric = {
        _sanitize(k): v for k, v in metrics.items() if isinstance(v, (int, float))
    }
    
    if numeric:
        mlflow.log_metrics(numeric)
        
def get_best_run(stage: str, recall_at_k: str = "recall_at_5", asc: bool = False) -> dict:

    runs = mlflow.search_runs(
        experiment_names=[f"{settings.mlflow_experiment_prefix}-{stage}"],
        order_by=[f"metrics.{recall_at_k} {'asc' if asc else 'desc'}"],
        max_results=1
    )

    if runs.empty:
        raise ValueError(f"No runs found for stage: {stage}")

    return runs.iloc[0].to_dict()

@contextmanager
def tracked_run(stage: str, run_name: str, params: dict):

    mlflow.set_experiment(f"{settings.mlflow_experiment_prefix}-{stage}")

    with mlflow.start_run(run_name=run_name) as run:
        
        mlflow.log_params(params)
        yield run
