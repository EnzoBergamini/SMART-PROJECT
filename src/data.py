from picsellia import Client  # type: ignore
from picsellia.types.enums import AnnotationFileType  # type: ignore

import zipfile
import shutil
import yaml  # type: ignore

from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split  # type: ignore

load_dotenv()


def data_extraction(
    piscellia_api_key: str, picsellia_dataset_uid: str, path: str
) -> None:
    """Extract the images and labels from the dataset version and save them in the data folder.

    Args:
        piscellia_api_key (str): The API key from Picsellia
        picsellia_dataset_uid (str): The dataset UID from Picsellia

    Returns:
        None
    """
    client = Client(api_token=piscellia_api_key)

    dataset_version = client.get_dataset_version_by_id(id=picsellia_dataset_uid)

    annotation_file_path = dataset_version.export_annotation_file(
        annotation_file_type=AnnotationFileType.YOLO, target_path=path
    )

    with zipfile.ZipFile(annotation_file_path, "r") as zip_ref:
        zip_ref.extractall(path + "/labels")

    folder_to_remove = (
        annotation_file_path.split("/")[0] + "/" + annotation_file_path.split("/")[1]
    )

    shutil.rmtree(folder_to_remove)

    dataset_version.download(target_path=path + "/images")

    shutil.move(path + "/labels/data.yaml", path + "/data.yaml")
    print("Data extraction completed.")


def data_validation(path: str) -> None:
    """Validate the data extracted from Picsellia.

    Returns:
        None
    """
    if not os.path.exists(path + "/images"):
        raise FileNotFoundError("The images folders are missing.")

    if len(os.listdir(path + "/images")) == 0:
        raise FileNotFoundError("The images folders are empty.")

    if not os.path.exists(path + "/labels"):
        raise FileNotFoundError("The labels folders are missing.")

    if len(os.listdir(path + "/labels")) == 0:
        raise FileNotFoundError("The labels folders are empty.")

    if len(os.listdir(path + "/images")) != len(os.listdir(path + "/labels")):
        raise ValueError("The number of images and labels are different.")

    image_files = set(os.path.splitext(f)[0] for f in os.listdir(path + "/images"))
    label_files = set(os.path.splitext(f)[0] for f in os.listdir(path + "/labels"))

    if not image_files == label_files:
        raise ValueError("Mismatch between image files and label files.")

    print("Data validation completed.")


def data_preparation(
    random_seed: int, test_size: float, val_size: float, path: str
) -> None:
    """Prepare the data by splitting it into training, validation, and testing sets.

    Args:
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the training dataset to include in the validation split.
        random_seed (int): The seed used by the random number generator.

    Returns:
        None
    """
    image_files = os.listdir(path + "/images")
    label_files = os.listdir(path + "/labels")

    image_files.sort()
    label_files.sort()

    images_train, images_test, labels_train, labels_test = train_test_split(
        image_files, label_files, test_size=test_size, random_state=random_seed
    )

    val_size = val_size / (1 - test_size)

    images_train, images_val, labels_train, labels_val = train_test_split(
        images_train, labels_train, test_size=val_size, random_state=random_seed
    )

    os.makedirs(path + "/images/train", exist_ok=True)
    os.makedirs(path + "/labels/train", exist_ok=True)

    for image, label in zip(images_train, labels_train):
        shutil.move(f"{path}/images/{image}", path + "/images/train")
        shutil.move(f"{path}/labels/{label}", path + "/labels/train")

    os.makedirs(path + "/images/test", exist_ok=True)
    os.makedirs(path + "/labels/test", exist_ok=True)

    for image, label in zip(images_test, labels_test):
        shutil.move(f"{path}/images/{image}", path + "/images/test")
        shutil.move(f"{path}/labels/{label}", path + "/labels/test")

    os.makedirs(path + "/images/val", exist_ok=True)
    os.makedirs(path + "/labels/val", exist_ok=True)

    for image, label in zip(images_val, labels_val):
        shutil.move(f"{path}/images/{image}", path + "/images/val")
        shutil.move(f"{path}/labels/{label}", path + "/labels/val")

    with open(path + "/data.yaml", "r") as file:
        data = yaml.safe_load(file)

    yolo_data = {
        "path": path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(data["names"])},
    }

    with open(path + "/config.yaml", "w") as file:
        yaml.dump(yolo_data, file, default_flow_style=False, allow_unicode=True)
        os.remove(path + "/data.yaml")

    print("Data preparation completed.")
