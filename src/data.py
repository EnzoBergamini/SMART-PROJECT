from picsellia import Client  # type: ignore
from picsellia.types.enums import AnnotationFileType  # type: ignore

import zipfile
import shutil

from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split  # type: ignore

load_dotenv()


def data_extraction(piscellia_api_key: str, picsellia_dataset_uid: str) -> None:
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
        annotation_file_type=AnnotationFileType.YOLO, target_path="data/"
    )

    with zipfile.ZipFile(annotation_file_path, "r") as zip_ref:
        zip_ref.extractall("data/labels")

    folder_to_remove = (
        annotation_file_path.split("/")[0] + "/" + annotation_file_path.split("/")[1]
    )

    shutil.rmtree(folder_to_remove)

    dataset_version.download(target_path="data/images")

    shutil.move("data/labels/data.yaml", "data/data.yaml")


def data_validation() -> None:
    """Validate the data extracted from Picsellia.

    Returns:
        None
    """
    if not os.path.exists("data/images"):
        raise FileNotFoundError("The images folders are missing.")

    if len(os.listdir("data/images")) == 0:
        raise FileNotFoundError("The images folders are empty.")

    if not os.path.exists("data/labels"):
        raise FileNotFoundError("The labels folders are missing.")

    if len(os.listdir("data/labels")) == 0:
        raise FileNotFoundError("The labels folders are empty.")

    if len(os.listdir("data/images")) != len(os.listdir("data/labels")):
        raise ValueError("The number of images and labels are different.")

    image_files = set(os.path.splitext(f)[0] for f in os.listdir("data/images"))
    label_files = set(os.path.splitext(f)[0] for f in os.listdir("data/labels"))

    if not image_files == label_files:
        raise ValueError("Mismatch between image files and label files.")


def data_preparation(random_seed: int, test_size: float, val_size: float) -> None:
    """Prepare the data by splitting it into training, validation, and testing sets.

    Args:
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the training dataset to include in the validation split.
        random_seed (int): The seed used by the random number generator.

    Returns:
        None
    """
    image_files = os.listdir("data/images")
    label_files = os.listdir("data/labels")

    image_files.sort()
    label_files.sort()

    images_train, images_test, labels_train, labels_test = train_test_split(
        image_files, label_files, test_size=test_size, random_state=random_seed
    )

    val_size = val_size / (1 - test_size)

    images_train, images_val, labels_train, labels_val = train_test_split(
        images_train, labels_train, test_size=val_size, random_state=random_seed
    )

    os.makedirs("data/train/images", exist_ok=True)
    os.makedirs("data/train/labels", exist_ok=True)

    for image, label in zip(images_train, labels_train):
        shutil.move(f"data/images/{image}", "data/train/images")
        shutil.move(f"data/labels/{label}", "data/train/labels")

    os.makedirs("data/test/images", exist_ok=True)
    os.makedirs("data/test/labels", exist_ok=True)

    for image, label in zip(images_test, labels_test):
        shutil.move(f"data/images/{image}", "data/test/images")
        shutil.move(f"data/labels/{label}", "data/test/labels")

    os.makedirs("data/val/images", exist_ok=True)
    os.makedirs("data/val/labels", exist_ok=True)

    for image, label in zip(images_val, labels_val):
        shutil.move(f"data/images/{image}", "data/val/images")
        shutil.move(f"data/labels/{label}", "data/val/labels")

    shutil.rmtree("data/images")
    shutil.rmtree("data/labels")
