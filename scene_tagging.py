import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import json

# --- CONFIG ---
photo_dir = "../photoprism/originals/flickr30k_images"
output_file = "scene_tags.json"
model_path = "models/places365/resnet18_places365.pth.tar"

# --- Load Model ---
def load_model():
    model = models.resnet18(num_classes=365)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- Load Labels ---
classes = [line.strip().split(' ')[0][3:] for line in open("models/places365/categories_places365.txt")]

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

model = load_model()
scene_tags = {}

# --- Tag Photos ---
for fname in os.listdir(photo_dir):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    path = os.path.join(photo_dir, fname)
    try:
        img = Image.open(path).convert('RGB')
        input_img = transform(img).unsqueeze(0)
        with torch.no_grad():
            logit = model(input_img)
            probs = torch.nn.functional.softmax(logit, 1)
            top5 = torch.topk(probs, 5)
        scene_list = [classes[i] for i in top5.indices[0]]
        scene_tags[fname] = scene_list
        print(f"{fname}: {scene_list}")
    except Exception as e:
        print(f"Failed to process {fname}: {e}")

# --- Save Results ---
with open(output_file, "w") as f:
    json.dump(scene_tags, f, indent=2)
print(f"\nSaved scene tags to {output_file}")
