import os
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
from tqdm import tqdm
import shutil

# --- Config ---
face_dir = "face_thumbnails"
output_dir = "face_clusters"
os.makedirs(output_dir, exist_ok=True)

# --- Step 1: Load and Encode All Faces ---
encodings = []
file_names = []

for file in tqdm(os.listdir(face_dir), desc="Encoding faces"):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(face_dir, file)
    img = face_recognition.load_image_file(path)
    faces = face_recognition.face_encodings(img)
    if faces:
        encodings.append(faces[0])
        file_names.append(file)

# --- Step 2: Cluster ---
clustering = DBSCAN(metric="euclidean", eps=0.5, min_samples=2)
labels = clustering.fit_predict(encodings)

# --- Step 3: Organize by Cluster ---
for label, file in zip(labels, file_names):
    person_dir = os.path.join(output_dir, f"Person_{label}" if label != -1 else "Unknown")
    os.makedirs(person_dir, exist_ok=True)
    shutil.copy(os.path.join(face_dir, file), os.path.join(person_dir, file))

print(f"\n✅ Grouped {len(file_names)} faces into {len(set(labels)) - (1 if -1 in labels else 0)} people.")
print(f"📂 Output saved to: {output_dir}/")
