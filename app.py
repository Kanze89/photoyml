import streamlit as st
import os
from PIL import Image

# --- Config ---
face_root = "../face_clusters"
photo_root = "../photoprism/originals/flickr30k_images"

st.set_page_config(page_title="Face Browser", layout="wide")
st.title("🧠 Face Cluster Browser")

# List all groups
groups = sorted([g for g in os.listdir(face_root) if os.path.isdir(os.path.join(face_root, g))])

if not groups:
    st.warning("⚠️ No face clusters found in 'face_clusters/'.")
    st.stop()

# Select a person group
selected_group = st.sidebar.selectbox("👤 Select a Person Group", groups)

# --- Display face thumbnails ---
st.subheader(f"📂 {selected_group}")
face_dir = os.path.join(face_root, selected_group)
thumbs = [f for f in os.listdir(face_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

cols = st.columns(5)
for i, fname in enumerate(thumbs):
    col = cols[i % 5]
    try:
        img = Image.open(os.path.join(face_dir, fname))
        col.image(img, caption=fname, use_container_width=True)
    except:
        col.warning(f"Could not load: {fname}")

# --- Show original images checkbox ---
st.markdown("---")
if st.checkbox("🖼️ Show original photos containing this person"):
    st.subheader("Original Photo Matches")
    orig_files = sorted(set([f.split("_face_")[0] + ".jpg" for f in thumbs]))
    cols = st.columns(4)
    for i, of in enumerate(orig_files):
        full_path = os.path.join(photo_root, of)
        if os.path.exists(full_path):
            col = cols[i % 4]
            try:
                img = Image.open(full_path)
                col.image(img, caption=of, use_container_width=True)
            except:
                col.warning(f"Could not open original: {of}")
