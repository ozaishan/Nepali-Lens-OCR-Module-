from ultralytics import YOLO
import cv2
import os

class TextDetector:
    def __init__(self, model_path: str = None):

        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded YOLOv8 model from {model_path}")

    def train(self, data_yaml: str, epochs: int = 50, imgsz: int = 640, weights: str = 'yolov8n.pt', name: str = 'exp_text_detect'):
 
        self.model = YOLO(weights)  # base model
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            save=True,
            name=name
        )
        
        best_path = os.path.join("runs", "detect", name, "weights", "best.pt")
        print("✅ Training completed.")
        print(f"📁 Best weights saved at: {best_path}")
        return best_path  

    def predict(self, image_path: str, save_crops: bool = False, crop_dir: str = None, return_with_boxes: bool = False):
        """
        Run detection.
        - returns list of crop images (numpy arrays)
        - if save_crops: saves each crop to crop_dir
        - if return_with_boxes: returns (crops, boxes) tuple
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call train() or pass a valid model_path on init.")
        
        results = self.model(image_path)[0]  # process first image
        image = cv2.imread(image_path)
        crops = []
        boxes = []
        
        for i, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.tolist())
            crop = image[y1:y2, x1:x2]
            crops.append(crop)
            boxes.append((x1, y1, x2, y2))
            
            if save_crops and crop_dir:
                os.makedirs(crop_dir, exist_ok=True)
                out_path = os.path.join(crop_dir, f"crop_{i:03d}.png")
                cv2.imwrite(out_path, crop)
        
        if return_with_boxes:
            return crops, boxes
        else:
            return crops
