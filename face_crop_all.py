import os
import face_recognition
from PIL import Image
from tqdm import tqdm

# --- Config ---
image_folder = "photoprism/originals/flickr30k_images"
output_folder = "face_thumbnails"
os.makedirs(output_folder, exist_ok=True)

# --- Process ---
images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for fname in tqdm(images, desc="Cropping faces"):
    path = os.path.join(image_folder, fname)
    try:
        img = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(img)
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_image = img[top:bottom, left:right]
            pil_img = Image.fromarray(face_image)
            output_name = f"{os.path.splitext(fname)[0]}_face_{i+1}.jpg"
            output_path = os.path.join(output_folder, output_name)
            pil_img.save(output_path)
    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

print(f"\n✅ Done! All faces saved in '{output_folder}'")
