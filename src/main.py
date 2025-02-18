from data import data_extraction, data_preparation, data_validation
from training import model_training
from config import settings


if __name__ == "__main__":
    # DATA PIPELINE
    data_extraction(
        piscellia_api_key=settings.PICSELLIA_API_KEY,
        picsellia_dataset_uid=settings.PICSELLIA_DATASET_UID,
        path=settings.DATA_PATH,
    )

    data_validation(path=settings.DATA_PATH)

    data_preparation(
        random_seed=settings.DATA_SPLIT_SEED,
        test_size=settings.TEST_SIZE,
        val_size=settings.VAL_SIZE,
        path=settings.DATA_PATH,
    )

    # TRAINING PIPELINE
    model_training(
        pretrained_model_name=settings.PRETRAINED_MODEL_NAME,
        data_yaml_path=settings.TRAIN_DATA_YAML_PATH,
        epochs=settings.TRAIN_EPOCHS,
        batch=settings.TRAIN_BATCH_SIZE,
        learning_rate=settings.TRAIN_LEARNING_RATE,
        seed=settings.TRAIN_SEED,
        optimizer=settings.TRAIN_OPTIMIZER,
        device=settings.TRAIN_DEVICE,
    )
