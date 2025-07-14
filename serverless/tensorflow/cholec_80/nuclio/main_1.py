# import json
# import base64
# import io
# import numpy as np
# from PIL import Image
# import cv2
# from ultralytics import YOLO

# def init_context(context):
#     context.logger.info("Initializing YOLOv8 segmentation interactor")
#     context.user_data.model = YOLO("/opt/nuclio/best.pt")
#     context.user_data.labels = {
#         0: "Black Background", 1: "Abdominal Wall", 2: "Liver",
#         3: "Gastrointestinal Tract", 4: "Fat", 5: "Grasper",
#         6: "Connective Tissue", 7: "Blood", 8: "Cystic Duct",
#         9: "L-hook Electrocautery", 10: "Gallbladder",
#         11: "Hepatic Vein", 12: "Liver Ligament"
#     }
#     context.logger.info("Model loaded")

# def handler(context, event):
#     body = event.body  # dict
#     img = Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB")
#     points = body.get("pos_points", [])
#     threshold = float(body.get("threshold", 0.5))

#     # Применяем модель сегментации к изображению целиком.
#     results = context.user_data.model.predict(
#         source=np.array(img),
#         conf=threshold,
#         task="segment"
#     )
#     seg = results[0]
#     masks = seg.masks.data.cpu().numpy()
#     boxes = seg.boxes.xyxy.cpu().numpy()
#     scores = seg.boxes.conf.cpu().numpy()
#     classes = seg.boxes.cls.cpu().numpy().astype(int)

#     output = []
#     for i, mask in enumerate(masks):
#         mask_uint8 = (mask * 255).astype(np.uint8)
#         mask_uint8 = cv2.dilate(mask_uint8, np.ones((3, 3), np.uint8), iterations=1)
#         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         valid_polygons = []
#         for cnt in contours:
#             pts = cnt.squeeze()
#             if pts.ndim == 2 and pts.shape[0] >= 3:
#                 poly = pts.flatten().tolist()
#                 valid_polygons.append(poly)
#         context.logger.info(f"Contours: {len(contours)}, valid polygons: {len(valid_polygons)}")

#         if not valid_polygons:
#             continue

#         x1, y1, x2, y2 = boxes[i].tolist()
#         label = context.user_data.labels.get(classes[i], "unknown")
#         output.append({
#             "label": label,
#             "confidence": float(scores[i]),
#             "bounding_box": [x1, y1, x2, y2],
#             "segmentation": valid_polygons,
#             "type": "polygon"
#         })

#     return context.Response(
#         body=json.dumps(output),
#         headers={"Content-Type": "application/json"},
#         status_code=200
#     )


# import json
# import base64
# from PIL import Image
# import io
# import os
# import numpy as np
# import cv2
# from ultralytics import YOLO

# def init_context(context):
#     context.logger.info("Инициализация контекста... 0%")
#     model = ModelHandler()
#     context.user_data.model = model
#     context.logger.info("Инициализация контекста...100%")

# class ModelHandler:
#     def __init__(self):
#         model_path = os.environ.get("MODEL_PATH", "/opt/nuclio/best.pt")
#         self.min_points = int(os.environ.get("MIN_POINTS", 3))
#         self.max_points = int(os.environ.get("MAX_POINTS", 10))
#         self.model = YOLO(model_path)

#     def handle(self, image: Image.Image, points: list) -> list:
#         if not isinstance(points, list):
#             raise ValueError(f"Expected list of points, got {type(points)}")
#         if len(points) < self.min_points or len(points) > self.max_points:
#             raise ValueError(
#                 f"Number of points must be between {self.min_points} and {self.max_points}, got {len(points)}"
#             )

#         numpy_image = np.array(image.convert("RGB"))
#         height, width = numpy_image.shape[:2]
#         points_arr = np.asarray(points, dtype=int)

#         valid_mask = (points_arr[:, 0] >= 0) & (points_arr[:, 0] < width) & \
#                      (points_arr[:, 1] >= 0) & (points_arr[:, 1] < height)
#         if not valid_mask.all():
#             invalid = points_arr[~valid_mask]
#             raise IndexError(f"Points out of image bounds: {invalid.tolist()}")

#         results = self.model(numpy_image, device='cpu', verbose=False)
#         masks = getattr(results[0].masks, 'data', None)
#         if masks is None or len(masks) == 0:
#             raise RuntimeError("Маски не обнаружены")

#         selected_mask = None
#         max_area = 0
#         for mask in masks:
#             mask_bool = mask.cpu().numpy().astype(bool)
#             if all(mask_bool[pt[1], pt[0]] for pt in points_arr):
#                 area = mask_bool.sum()
#                 if area > max_area:
#                     max_area = area
#                     selected_mask = mask_bool
#         if selected_mask is None:
#             raise RuntimeError("Не удалось сопоставить точки ни с одной маской")

#         mask_uint8 = (selected_mask.astype(np.uint8) * 255)
#         contours_info = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
#         main_contour = max(contours, key=lambda cnt: cnt.shape[0])
#         polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in main_contour]
#         return polygon

# def handler(context, event):
#     context.logger.info("Вызов handler...")
#     data = event.body
#     points = data.get("pos_points", [])
#     image_data = data.get("image", "")
#     buf = io.BytesIO(base64.b64decode(image_data))
#     image = Image.open(buf)
#     polygon = context.user_data.model.handle(image, points)
#     return context.Response(
#         body=json.dumps(polygon),
#         headers={},
#         content_type='application/json',
#         status_code=200
#     )


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

def init_context(context):
    # Настройка логера
    logging.basicConfig(level=logging.INFO)
    context.logger.info("Инициализация interactor для CVAT начата")
    # Инициализация обработчика модели
    context.user_data.model = SegmentationInteractor(
        model_path=os.environ.get("MODEL_PATH", "/opt/nuclio/best.pt"),
        min_points=int(os.environ.get("MIN_POINTS", 3)),
        max_points=int(os.environ.get("MAX_POINTS", 10)),
        default_conf=float(os.environ.get("DEFAULT_CONF", 0.5)),
    )
    context.logger.info("Инициализация interactor для CVAT завершена")

class SegmentationInteractor:
    def __init__(
        self,
        model_path: str,
        min_points: int = 3,
        max_points: int = 10,
        default_conf: float = 0.5
    ):
        self.min_points = min_points
        self.max_points = max_points
        self.default_conf = default_conf
        self.model = YOLO(model_path)

    def _validate_points(self, points: List[List[int]], width: int, height: int):
        if not isinstance(points, list):
            raise ValueError(f"Ожидается список точек, получено: {type(points)}")
        count = len(points)
        if count < self.min_points or count > self.max_points:
            raise ValueError(
                f"Количество точек должно быть между {self.min_points} и {self.max_points}, получено: {count}"
            )
        for idx, (x, y) in enumerate(points):
            if not (0 <= x < width and 0 <= y < height):
                raise IndexError(f"Точка {idx} с координатами {(x, y)} выходит за границы изображения {width}x{height}")

    def handle(
        self,
        image: Image.Image,
        points: List[List[int]],
        conf: float = None
    ) -> List[List[int]]:
        conf = conf if isinstance(conf, float) else self.default_conf
        img_array = np.array(image.convert("RGB"))
        height, width = img_array.shape[:2]
        self._validate_points(points, width, height)

        # Запуск сегментации
        results = self.model.predict(
            source=img_array,
            conf=conf,
            task="segment",
            verbose=False
        )
        if not results or not results[0].masks:
            raise RuntimeError("Сегментация не обнаружила маски с заданным порогом уверенности")

        masks = results[0].masks.data.cpu().numpy()
        selected = None
        max_area = 0
        for mask in masks:
            mask_bool = np.array(mask, dtype=bool)
            try:
                mh, mw = mask_bool.shape
                scale_x = mw / width
                scale_y = mh / height
                # Приведение координат к масштабу маски
                if all(mask_bool[int(y * scale_y), int(x * scale_x)] for x, y in points):
                    area = mask_bool.sum()
                    if area > max_area:
                        max_area = area
                        selected = mask_bool
            except Exception as e:
                raise RuntimeError(f"Ошибка при проверке маски: {e}")

        if selected is None:
            raise RuntimeError("Ни одна маска не содержит все заданные точки")

        # Поиск контура и сбор полигона
        mask_uint8 = (selected.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("Контур маски не найден")
        main_contour = max(contours, key=lambda c: cv2.contourArea(c))
        # Преобразование контура в список координат
        polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in main_contour]
        return polygon


def handler(context, event):
    context.logger.info("Получен запрос от CVAT interactor")
    body = event.body
    try:
        image_data = body.get("image", "")
        pos_points = body.get("pos_points", [])
        conf_val = float(body.get("threshold", context.user_data.model.default_conf))

        # Декодирование и открытие изображения
        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes))

        # Обработка изображения и точек
        polygon = context.user_data.model.handle(image, pos_points, conf_val)
        # response_body = json.dumps({"polygon": polygon})
        response_body = json.dumps(polygon)
        status = 200
    except Exception as e:
        context.logger.error(f"Ошибка обработки: {e}")
        response_body = json.dumps({"error": str(e)})
        status = 400

    context.logger.info(f"Высылается результат: {response_body}")
    return context.Response(
        body=response_body,
        headers={"Content-Type": "application/json"},
        status_code=status
    )
