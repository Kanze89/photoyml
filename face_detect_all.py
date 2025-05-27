import os
import face_recognition
import json
from PIL import Image
from tqdm import tqdm

# --- Config ---
image_folder = "photoprism/originals/flickr30k_images"
output_file = "faces_detected.json"

# --- Process ---
results = {}
images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for fname in tqdm(images, desc="Detecting faces"):
    path = os.path.join(image_folder, fname)
    try:
        image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(image)
        results[fname] = len(face_locations)
    except Exception as e:
        print(f"❌ Error on {fname}: {e}")

# --- Save ---
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Face detection done. Results saved to {output_file}")
