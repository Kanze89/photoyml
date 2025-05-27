import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import json

# --- Configuration ---
image_folder = "photoprism/originals/flickr30k_images"
output_file = "captions_blip.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load BLIP Model ---
print("🔄 Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# --- Gather Images ---
captions = {}
images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# --- Caption Loop ---
for fname in tqdm(images, desc="🧠 Generating captions"):
    path = os.path.join(image_folder, fname)
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions[fname] = caption
    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

# --- Save Output ---
with open(output_file, "w") as f:
    json.dump(captions, f, indent=2)

print(f"\n✅ Done! Captions saved to: {output_file}")
