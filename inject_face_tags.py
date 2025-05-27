import os
import subprocess
from tqdm import tqdm

# --- CONFIG ---
face_clusters_dir = "face_clusters"
originals_dir = "photoprism/originals/flickr30k_images"

# Map: original file → set of person labels
photo_tags = {}

for group in os.listdir(face_clusters_dir):
    group_path = os.path.join(face_clusters_dir, group)
    if not os.path.isdir(group_path):
        continue
    for filename in os.listdir(group_path):
        if "_face_" not in filename:
            continue
        # Extract base photo name
        original_file = filename.split("_face_")[0] + ".jpg"
        photo_tags.setdefault(original_file, set()).add(group)

# --- Inject EXIF keywords ---
for photo, tags in tqdm(photo_tags.items(), desc="Tagging photos"):
    full_path = os.path.join(originals_dir, photo)
    if not os.path.exists(full_path):
        continue
    keywords = " ".join([f"-Keywords+={tag}" for tag in tags])
    cmd = f"exiftool -overwrite_original {keywords} '{full_path}'"
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\n✅ EXIF tags added. You can now search by person name in PhotoPrism!")
