import mlflow
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from model import SKLearnModel
from preprocessing import preprocess_data


def train(
    experiment_name: str,
    model: SKLearnModel,
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Main training pipeline.

    Args:
        experiment_name (str): Name of MLFlow experiment.
        model (SKLearnModel): Wrapped SKLearn model.
        df (pd.DataFrame): Pandas dataframe holding data for prediction.
        target_column (str): Name of column to use for predition.
        test_size (float, optional): Proportion of test. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
    """
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(
        df, target_column, test_size, random_state
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model.model)])

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", report.pop("accuracy"))

        for key in report:
            mlflow.log_metrics(
                {
                    f"{key}_precision": report[key]["precision"],
                    f"{key}_recall": report[key]["recall"],
                    f"{key}_f1_score": report[key]["f1-score"],
                }
            )

        mlflow.sklearn.log_model(pipeline, "model", input_example=X_train.iloc[:5])
