import json
import os

# Input files
places_file = "scene_tags.json"
clip_file = "clip_scene_tags.json"
output_file = "scene_combined.json"

# Load files
with open(places_file) as f:
    places_tags = json.load(f)

with open(clip_file) as f:
    clip_tags = json.load(f)

# Merge them
combined = {}

all_files = set(places_tags.keys()) | set(clip_tags.keys())

for fname in all_files:
    combined[fname] = []

    if fname in places_tags:
        combined[fname].extend(places_tags[fname])
    if fname in clip_tags:
        combined[fname].extend(clip_tags[fname])

    # Remove duplicates
    combined[fname] = list(set(combined[fname]))

# Save
with open(output_file, "w") as f:
    json.dump(combined, f, indent=2)

print(f"✅ Combined scene tags saved to {output_file}")
