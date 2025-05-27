import os
import clip
import torch
from PIL import Image
from tqdm import tqdm
import json

# --- CONFIG ---
image_folder = "photoprism/originals/flickr30k_images"
output_file = "captions_clip.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- INIT ---
model, preprocess = clip.load("ViT-B/32", device=device)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Example prompts CLIP compares against
prompts = [
    "a person", "a mountain", "a sunset", "a group of people",
    "a man on a horse", "a woman smiling", "a forest", "a dog running",
    "a beach", "a city street", "a family", "a child", "a bicycle"
]

text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)

# --- PROCESS ---
captions = {}
for file in tqdm(image_files, desc="Captioning images"):
    path = os.path.join(image_folder, file)
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            similarity = (image_features @ text_features.T).squeeze(0)
            best = similarity.argmax().item()
            captions[file] = prompts[best]
    except Exception as e:
        print(f"Error processing {file}: {e}")

# --- SAVE ---
with open(output_file, "w") as f:
    json.dump(captions, f, indent=2)

print(f"\n✅ Done! Captions saved to {output_file}")
