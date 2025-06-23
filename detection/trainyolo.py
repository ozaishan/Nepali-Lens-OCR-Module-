from text_detection import TextDetector

def main():
    det = TextDetector()
    best_weights = det.train(
        data_yaml="dataset/yolo/hwnepali.yaml",
        epochs=50,
        imgsz=640,
        weights="yolov8n.pt",
        name="exp_text_detect3"
    )
    
    # Load trained model for prediction
    det = TextDetector(model_path=best_weights)
    crops = det.predict("sample.jpg", save_crops=True, crop_dir="detected_texts")

if __name__ == "__main__":
    main()
