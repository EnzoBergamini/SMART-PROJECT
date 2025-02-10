from picsellia import Client  # type: ignore
from picsellia.types.enums import AnnotationFileType  # type: ignore

import zipfile
import shutil

from dotenv import load_dotenv
import os

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
