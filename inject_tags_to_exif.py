import os
import json
import subprocess
from tqdm import tqdm

# --- Paths ---
image_dir = "photoprism/originals/flickr30k_images"
caption_file = "captions_blip.json"
detection_file = "detections_yolo.json"

# --- Load Metadata ---
captions = json.load(open(caption_file)) if os.path.exists(caption_file) else {}
detections = json.load(open(detection_file)) if os.path.exists(detection_file) else {}

# --- Process Each Image ---
for fname in tqdm(os.listdir(image_dir), desc="Injecting tags"):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    tags = detections.get(fname, [])
    caption = captions.get(fname)

    if not tags and not caption:
        continue

    args = ["exiftool", "-overwrite_original"]
    
    # Add tags as EXIF keywords
    for tag in tags:
        args.append(f"-Keywords+={tag}")
    
    # Add caption as EXIF title or description
    if caption:
        args.append(f"-Title={caption}")
        args.append(f"-ImageDescription={caption}")
    
    args.append(os.path.join(image_dir, fname))
    
    subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
