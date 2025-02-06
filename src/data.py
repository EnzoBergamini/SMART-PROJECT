from picsellia import Client  # type: ignore
from picsellia.types.enums import AnnotationFileType  # type: ignore

import zipfile
import shutil
import os

from dotenv import load_dotenv

load_dotenv()


def data_extraction() -> None:
    """Extract the images and labels from the dataset version and save them in the data folder."""
    client = Client(api_token=os.getenv("PICSELLIA_API_KEY"))

    dataset_version = client.get_dataset_version_by_id(
        id=os.getenv("PICSELLIA_DATASET_UID")
    )

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


def data_validation() -> None:
    """Check if the data is correctly extracted."""
    if not os.path.exists("data/images") or not os.path.exists("data/labels"):
        raise FileNotFoundError("The data is not correctly extracted.")
    if len(os.listdir("data/images")) != len(os.listdir("data/labels")):
        raise ValueError("The number of images and labels is not the same.")
