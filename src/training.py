from ultralytics import YOLO  # type: ignore
import mlflow  # type: ignore
from mlflow.tracking import MlflowClient  # type: ignore
import time


def model_training(
    pretrained_model_name: str,
    data_yaml_path: str,
    epochs: int,
    batch: int,
    learning_rate: float,
    seed: int,
    optimizer: str,
    device: str,
) -> None:
    """Train the model using the training data.

    Args:
        model_name (str): The name of the model to train.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The number of samples per batch.
        learning_rate (float): The learning rate for the optimizer.
        seed (int): The seed used by the random number generator.
        optimizer (str): The optimizer to use for training.
        device (str): The device to use for training.

    Returns:
        None
    """
    with mlflow.start_run(
        run_name=pretrained_model_name + "-" + str(time.strftime("%d/%m/%Y-%H_%M_%S")),
        log_system_metrics=True,
    ):
        model = YOLO(pretrained_model_name)

        model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch,
            seed=seed,
            optimizer=optimizer,
            lr0=learning_rate,
            device=device,
        )

        mlflow.log_artifacts("../requirements.txt")


def model_validation(model_name: str) -> None:
    """Evaluate the trained model and manage MLflow model registry.

    1. Register the latest trained model.
    2. Compare it with the current "Champion" model.
    3. Promote the best model as the "Champion" and set the previous as "Challenger".

    Args:
        model_name (str): The name of the model in the MLflow Model Registry.
    """

    client = MlflowClient()

    last_run = mlflow.last_active_run()
    if last_run is None:
        raise RuntimeError("No active run found.")
    run_id = last_run.info.run_id

    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/weights/best.pt", name=model_name
    )

    new_mAP50 = client.get_metric_history(run_id, "mAP50")[-1].value

    try:
        champion_model = client.get_model_version_by_alias(model_name, "Champion")
        champion_mAP50 = client.get_metric_history(champion_model.run_id, "mAP50")[
            -1
        ].value
    except Exception:
        champion_model = None
        champion_mAP50 = 0

    if new_mAP50 > champion_mAP50:
        client.set_registered_model_alias(
            model_name, alias="Champion", version=model_version.version
        )

        if champion_model:
            client.set_registered_model_alias(
                model_name, alias="Challenger", version=champion_model.version
            )

        print(
            f"🎉 New Champion Model: {model_name} v{model_version.version} (mAP50={new_mAP50})"
        )
    else:
        print(
            f"🔬 Challenger Model: {model_name} v{model_version.version} (mAP50={new_mAP50})"
        )

    print("✅ Model validation and registry update completed!")
