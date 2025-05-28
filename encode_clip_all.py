import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_dir = "photoprism/originals/flickr30k_images"
output_file = "clip_embeddings.json"

results = {}

for fname in tqdm(sorted(os.listdir(image_dir)), desc="Encoding images"):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    try:
        path = os.path.join(image_dir, fname)
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().numpy()[0]
        results[fname] = embedding.tolist()
    except Exception as e:
        print(f"Failed {fname}: {e}")

with open(output_file, "w") as f:
    json.dump(results, f)

print(f"✅ Done! Saved to {output_file}")
