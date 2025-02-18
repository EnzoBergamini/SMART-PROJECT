import logging
import bentoml
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

logger=logging.getLogger("bentoml")
logger.setLevel(logging.DEBUG)

@bentoml.service
class YOLOService:
    def __init__(self):
        model_name = os.getenv("MODEL_NAME")
        self.model = self.load_model(model_name)

    def load_model(self, model_path:str) -> YOLO:
        return YOLO(model_path)
    
    @bentoml.api
    async def predict(self, image: Image.Image) -> dict:
        img_array = np.array(image)
        
        results = self.model.predict(img_array)
        result = results[0]

        boxes = []
        for box in result.boxes:
            boxes.append(
                {
                    "xyxy": box.xyxy[0].tolist(),
                    "class_id": int(box.cls[0])
                }
            )
        return {
            "boxes": boxes,
            "inference_time": float(result.speed["inference"])
        }