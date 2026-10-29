"""YOLO vehicle detection with a tiled fallback for small toy cars.

Notes:
- The default COCO-trained YOLO model was not trained specifically on toy cars.
- Larger inference size and overlapping crops can help when a toy car is tiny in the frame,
  but cannot guarantee detection. For dependable toy-car detection, fine-tune on labelled
  images of the actual toy cars.
"""
import os
import cv2
from ultralytics import YOLO

# COCO class IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def _env_int(name, default, minimum):
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name, default, minimum, maximum):
    try:
        return min(maximum, max(minimum, float(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


class ParkingDetector:
    def __init__(self, model_path=None, confidence=None):
        self.model_path = model_path or os.environ.get('YOLO_MODEL', 'yolov8n.pt')
        self.confidence = confidence if confidence is not None else _env_float('YOLO_CONFIDENCE', '0.12', 0.01, 0.99)
        self.image_size = _env_int('YOLO_IMGSZ', '640', 320)
        self.tile_image_size = _env_int('YOLO_TILE_IMGSZ', '768', 320)
        # Run the more expensive crop pass periodically, rather than on every empty frame.
        self.tile_every = _env_int('YOLO_TILE_EVERY', '2', 1)
        self.tile_cursor = 0
        self.model = YOLO(self.model_path)
        self.frame_count = 0
        self.last_boxes = []
        self.last_boxes_age = 999

    def _predict_boxes(self, image, offset_x=0, offset_y=0, imgsz=None):
        """Return vehicle boxes as (x1, y1, x2, y2, confidence, class_id)."""
        
        results = self.model.predict(
            image,
            conf=self.confidence,
            imgsz=imgsz or self.image_size,
            classes=sorted(VEHICLE_CLASS_IDS),
            max_det=100,
            device=0,
            verbose=False,
        )[0]
        

        found = []
        if results.boxes is None:
            return found
        ih, iw = image.shape[:2]
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            # Clamp to the crop before applying the crop's full-frame offset.
            x1 = max(0.0, min(float(iw - 1), x1)) + offset_x
            x2 = max(0.0, min(float(iw - 1), x2)) + offset_x
            y1 = max(0.0, min(float(ih - 1), y1)) + offset_y
            y2 = max(0.0, min(float(ih - 1), y2)) + offset_y
            found.append((x1, y1, x2, y2, float(box.conf[0].item()), cls_id))
        return found

    def _tiled_fallback(self, frame):
        """Check one overlapping crop per call to avoid periodic multi-second stalls."""
        height, width = frame.shape[:2]
        crop_w = min(width, max(320, int(width * 0.62)))
        crop_h = min(height, max(320, int(height * 0.62)))
        x_starts = sorted(set([0, max(0, width - crop_w)]))
        y_starts = sorted(set([0, max(0, height - crop_h)]))
        crops = [(x, y) for y in y_starts for x in x_starts]
        if not crops:
            return []
        x, y = crops[self.tile_cursor % len(crops)]
        self.tile_cursor += 1
        crop = frame[y:y + crop_h, x:x + crop_w]
        if crop.size == 0:
            return []
        return self._predict_boxes(crop, x, y, imgsz=self.tile_image_size)

    def _deduplicate(self, boxes):
        if not boxes:
            return []
        # OpenCV NMS removes duplicate detections created by overlapping crops.
        rects = []
        scores = []
        for x1, y1, x2, y2, conf, _cls in boxes:
            rects.append([int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))])
            scores.append(float(conf))
        indices = cv2.dnn.NMSBoxes(rects, scores, self.confidence, 0.45)
        if len(indices) == 0:
            return []
        return [boxes[int(i)] for i in indices.flatten()]

    def process(self, frame, slots):
        self.frame_count += 1
        # First try the whole image. Larger imgsz helps preserve details of small vehicles.
        boxes = self._predict_boxes(frame)

        # When the full-frame model finds nothing, inspect one crop at a time.
        # This improves tiny-object recall without running four expensive crops at once.
        if not boxes and self.frame_count % self.tile_every == 0:
            boxes = self._deduplicate(self._tiled_fallback(frame))
            if boxes:
                self.last_boxes = boxes
                self.last_boxes_age = 0

        if boxes:
            self.last_boxes = boxes
            self.last_boxes_age = 0
        else:
            self.last_boxes_age += 1
            # Keep a recent detection briefly to avoid flickering occupancy in a static scene.
            # Do not hold it for long if a vehicle has moved away.
            if self.last_boxes and self.last_boxes_age <= 2:
                boxes = self.last_boxes

        annotated = frame.copy()
        vehicles = []
        for x1f, y1f, x2f, y2f, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, (x1f, y1f, x2f, y2f))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            vehicles.append((x1, y1, x2, y2, cx, cy))
            label = f'{self.model.names.get(cls_id, "vehicle")} {conf:.2f}'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(annotated, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)

        occupied = []
        for slot in slots:
            x, y, w, h = int(slot['x']), int(slot['y']), int(slot['w']), int(slot['h'])
            is_occupied = False
            for vx1, vy1, vx2, vy2, cx, cy in vehicles:
                # Occupied if the vehicle center is in the slot OR boxes overlap materially.
                ix = max(0, min(x + w, vx2) - max(x, vx1))
                iy = max(0, min(y + h, vy2) - max(y, vy1))
                intersection = ix * iy
                vehicle_area = max(1, (vx2 - vx1) * (vy2 - vy1))
                if (x <= cx <= x + w and y <= cy <= y + h) or intersection / vehicle_area >= 0.20:
                    is_occupied = True
                    break
            slot_id = str(slot.get('id', 'Slot'))
            if is_occupied:
                occupied.append(slot_id)
            color = (0, 0, 255) if is_occupied else (255, 180, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, f'{slot_id}: {"Occupied" if is_occupied else "Available"}', (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated, len(vehicles), occupied
