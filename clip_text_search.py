import os
import clip
import torch
from PIL import Image

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess images
image_folder = "test-images"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

images = []
valid_files = []
for f in image_files:
    try:
        img = Image.open(f).convert("RGB")
        images.append(preprocess(img).unsqueeze(0))
        valid_files.append(f)
    except:
        print(f"Skipped: {f}")

image_input = torch.cat(images).to(device)

# Ask user for a search query
query = input("🔍 Enter your search text: ")
text = clip.tokenize([query]).to(device)

# Encode
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text)

    # Compute similarity
    similarities = (image_features @ text_features.T).squeeze(1)

# Sort results
sorted_indices = similarities.argsort(descending=True)
print("\n📸 Search Results (Top Matches):\n")
for idx in sorted_indices[:5]:  # Show top 5 results
    print(f"{valid_files[idx]} — score: {similarities[idx].item():.4f}")
