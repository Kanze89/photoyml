import os
import json
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# --- Config ---
image_folder = "photoprism/originals/flickr30k_images"
output_file = "detections_yolo.json"
model_name = "yolov8n.pt"  # small + fast; change to yolov8m.pt for more accuracy

# --- Load Model ---
model = YOLO(model_name)

# --- Run Detection ---
results = {}
images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for fname in tqdm(images, desc="Detecting objects"):
    path = os.path.join(image_folder, fname)
    try:
        r = model(path, verbose=False)[0]
        labels = [model.names[int(cls)] for cls in r.boxes.cls]
        results[fname] = list(set(labels))  # Remove duplicates
    except Exception as e:
        print(f"❌ Error on {fname}: {e}")

# --- Save Results ---
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Object detection done. Results saved to {output_file}")
