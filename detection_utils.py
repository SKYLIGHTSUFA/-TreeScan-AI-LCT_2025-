import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import os
from config import DET_PATH


class TreeDetector:
    def __init__(self, model_path=DET_PATH, conf_threshold=0.2):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_trees(self, image_path_or_pil):
        """
        Детектирует деревья на изображении
        Returns: список словарей с информацией о bounding boxes
        """
        if isinstance(image_path_or_pil, Image.Image):
            # Конвертируем PIL в numpy для YOLO
            image_np = np.array(image_path_or_pil)
            results = self.model(
                image_np, conf=self.conf_threshold, imgsz=image_np.shape[0]
            )
        else:
            results = self.model(
                image_path_or_pil, conf=self.conf_threshold, imgsz=image_np.shape[0]
            )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Координаты bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    detections.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(confidence),
                            "class_id": class_id,
                            "class_name": self.model.names[class_id],
                        }
                    )

        return detections

    def crop_bbox(self, image_pil, bbox):
        """Вырезает регион по bounding box из изображения"""
        x1, y1, x2, y2 = bbox
        return image_pil.crop((x1, y1, x2, y2))

    def draw_detections(self, image_pil, detections, analysis_results=None):
        """
        Рисует bounding boxes и информацию на изображении
        """
        image = image_pil.copy()
        draw = ImageDraw.Draw(image)

        # Простой шрифт (если доступен)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 0),
            (0, 128, 0),
        ]

        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            color = colors[i % len(colors)]

            # Рисуем bounding box
            draw.rectangle(bbox, outline=color, width=3)

            # Текст с информацией
            label = f"Tree {i+1} ({detection['confidence']:.2f})"
            if analysis_results and i < len(analysis_results):
                health = analysis_results[i].get("Оценка здоровья", "N/A")
                species = (
                    analysis_results[i]
                    .get("Вид и порода", "N/A")
                    .split(":")[-1]
                    .strip()[:20]
                )
                label = f"{i+1}: {species} H:{health}"

            # Фон для текста
            text_bbox = draw.textbbox((bbox[0], bbox[1] - 25), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((bbox[0], bbox[1] - 25), label, fill=(255, 255, 255), font=font)

        return image
