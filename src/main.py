from training import model_training


if __name__ == "__main__":
    # DATA PIPELINE
    # data_extraction(
    #     piscellia_api_key=settings.PICSELLIA_API_KEY,
    #     picsellia_dataset_uid=settings.PICSELLIA_DATASET_UID,
    #     path=settings.DATA_PATH,
    # )
    # data_validation(path=settings.DATA_PATH)
    # data_preparation(
    #     random_seed=settings.DATA_SPLIT_SEED,
    #     test_size=settings.TEST_SIZE,
    #     val_size=settings.VAL_SIZE,
    #     path=settings.DATA_PATH,
    # )

    model_training(
        pretrained_model_name="yolo11n.pt",
        data_yaml_path="data/config.yaml",
        epochs=2,
        batch=10,
        learning_rate=0.1,
        seed=42,
        optimizer="SGD",
        device="mps",
    )
