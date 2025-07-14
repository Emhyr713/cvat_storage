import os
import io
import json
import base64
import logging
from typing import List
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

import tensorflow as tf

import torch  # Для YOLO (Ultralytics использует PyTorch)
import tensorflow as tf

# Проверка для TensorFlow
print("TensorFlow GPU доступен:", tf.config.list_physical_devices('GPU'))

# Проверка для PyTorch (YOLO)
print("PyTorch GPU доступен:", torch.cuda.is_available())
print("PyTorch устройство по умолчанию:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def init_context(context):
    logging.basicConfig(level=logging.INFO)
    context.logger.info("Инициализация interactor для CVAT начата")
    context.user_data.model = PolygonDetector(
        model_path=os.environ.get("MODEL_PATH", "/opt/nuclio/best.pt"),
        default_conf=float(os.environ.get("DEFAULT_CONF", 0.3)),
    )
    context.logger.info("Инициализация interactor для CVAT завершена")

# Индивидуальные пороги доверия для каждой метки
LABEL_THRESHOLDS = {
    0: 0.4,   # Black Background
    1: 0.4,   # Abdominal Wall
    2: 0.45,   # Liver
    3: 0.4,   # Gastrointestinal Tract
    4: 0.45,  # Fat
    5: 0.6,   # Grasper
    6: 0.35,  # Connective Tissue
    7: 0.5,   # Blood
    8: 0.4,   # Cystic Duct
    9: 0.6,   # L-hook Electrocautery
    10: 0.4,  # Gallbladder
    11: 0.4,  # Hepatic Vein
    12: 0.4,  # Liver Ligament
}

MODEL_IMG_SIZE = 640
MIN_POLYGON_AREA = 100  # Минимальная площадь полигона в пикселях
CONTOUR_APPROX_EPSILON = 0.003  # Параметр для упрощения контуров

class PolygonDetector:
    def __init__(self, model_path: str, default_conf: float = 0.5):
        self.default_conf = default_conf
        # Явно указываем устройство (GPU, если доступен)
        self.device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
        self.model = YOLO(model_path).to(self.device)  # Перенос модели на GPU
        self.label_thresholds = LABEL_THRESHOLDS

        print("YOLO модель загружена на устройство:", next(self.model.model.parameters()).device)

    def handle(self, image: Image.Image, conf: float = None) -> List[dict]:
        conf = float(conf) if isinstance(conf, float) else self.default_conf
        img_array = np.array(image.convert("RGB"))
        orig_h, orig_w = img_array.shape[:2]

        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        resized_img = cv2.resize(img_array, (MODEL_IMG_SIZE, MODEL_IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        results = self.model.predict(
            source=resized_img,
            conf=conf,
            task="segment",
            verbose=False
        )

        if not results or results[0].masks is None:
            return []

        masks = results[0].masks.data.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        # masks = results[0].masks.data  # Не перемещаем на CPU сразу
        # scores = results[0].boxes.conf
        # classes = results[0].boxes.cls.int()


        labels = results[0].names

        scale_x = orig_w / MODEL_IMG_SIZE
        scale_y = orig_h / MODEL_IMG_SIZE

        annotations = []
        for i, mask in enumerate(masks):
            # Применяем индивидуальный порог для класса
            class_id = int(classes[i])
            class_threshold = self.label_thresholds.get(class_id, conf)
            if scores[i] < class_threshold:
                continue

            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Берем только самый большой контур
            contour = max(contours, key=cv2.contourArea)
            
            # Пропускаем маленькие полигоны
            if cv2.contourArea(contour) < MIN_POLYGON_AREA:
                continue

            # Упрощаем контур
            epsilon = CONTOUR_APPROX_EPSILON * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 3:
                continue

            polygon = approx.squeeze().tolist()
            if not isinstance(polygon[0], list):
                continue

            scaled_polygon = [
                [float(x * scale_x), float(y * scale_y)]
                for x, y in polygon
            ]
            flat_polygon = [float(coord) for point in scaled_polygon for coord in point]

            annotations.append({
                "confidence": str(float(scores[i])),
                "label": str(labels[class_id]),
                "points": flat_polygon,
                "type": "polygon"
            })

        return annotations


def handler(context, event):
    context.logger.info("Получен запрос от CVAT detector")
    body = event.body
    try:
        image_data = body.get("image", "")
        conf_val = float(body.get("threshold", context.user_data.model.default_conf))

        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes))

        predictions = context.user_data.model.handle(image, conf_val)
        response_body = json.dumps(predictions)
        status = 200
    except Exception as e:
        context.logger.error(f"Ошибка обработки: {e}")
        response_body = json.dumps({"error": str(e)})
        status = 400

    return context.Response(
        body=response_body,
        headers={"Content-Type": "application/json"},
        status_code=status
    )