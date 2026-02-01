from ultralytics import YOLO
import os

MODEL_PATH = "../models/yolov8.pt"
IMAGE_PATH = "input/test1.jpeg"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

results = model(
    IMAGE_PATH,
    save=True,
    project=OUTPUT_DIR,
    name="results",
    conf=0.25
)

print("Inference completed successfully!")
