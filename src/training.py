from ultralytics import YOLO  # type: ignore
import mlflow  # type: ignore
import time


def model_training(
    pretrained_model_name: str,
    data_yaml_path: str,
    epochs: int,
    batch_size: int,
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
        run_name=pretrained_model_name + "-" + str(time.strftime("%m/%d/%Y-%H_%M_%S")),
        log_system_metrics=True,
    ) as run:
        model = YOLO(pretrained_model_name)

        model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            optimizer=optimizer,
            lr0=learning_rate,
            device=device,
            save_dir="runs/",
        )

        mlflow.log_params(
            local_path="requirements.txt",
            artifact_path="environment",
            run_id=run.info.run_id,
        )

        params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "optimizer": optimizer,
        }

        mlflow.log_params(params=params, run_id=run.info.run_id)


def model_evaluation() -> None:
    """Evaluate the model using the test data.

    Returns:
        None
    """
    with mlflow.start_run(run_name="model_evaluation", log_system_metrics=True) as run:
        model = YOLO()
        model.test(data="data.yaml", batch_size=8, save_json=True)

        mlflow.log_params(
            local_path="requirements.txt",
            artifact_path="environment",
            run_id=run.info.run_id,
        )
