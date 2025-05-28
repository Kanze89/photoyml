import streamlit as st
import os
import json
from PIL import Image
from pathlib import Path

# --- Config ---
face_root = "../face_clusters"
photo_root = "../photoprism/originals/flickr30k_images"
caption_file = "../captions_blip.json"
detection_file = "../detections_yolo.json"

# --- Load Metadata ---
captions = json.load(open(caption_file)) if os.path.exists(caption_file) else {}
detections = json.load(open(detection_file)) if os.path.exists(detection_file) else {}

# --- UI Setup ---
st.set_page_config(page_title="📸 AI Photo Browser", layout="wide")
st.title("🧠 AI Photo Browser")
st.caption("Browse by face cluster, view original images, and explore AI-generated captions and tags.")

# --- Sidebar Person Group ---
groups = sorted([g for g in os.listdir(face_root) if os.path.isdir(os.path.join(face_root, g))])
selected_group = st.sidebar.selectbox("👤 Select a Person Group", groups)

# --- Face Thumbnails ---
st.subheader(f"Faces in {selected_group}")
thumbs = os.listdir(os.path.join(face_root, selected_group))
cols = st.columns(5)
for i, fname in enumerate(sorted(thumbs)):
    col = cols[i % 5]
    try:
        img = Image.open(os.path.join(face_root, selected_group, fname))
        col.image(img, caption=fname, use_container_width=True)
    except:
        col.warning(f"Error loading {fname}")

# --- Originals ---
st.markdown("---")
if st.checkbox("🖼️ Show original photos containing this person"):
    st.subheader("Original Photo Matches")
    orig_files = sorted(set([f.split("_face_")[0] + ".jpg" for f in thumbs]))

    for i, of in enumerate(orig_files):
        full_path = os.path.join(photo_root, of)
        if not os.path.exists(full_path):
            continue

        col1, col2 = st.columns([1, 3])
        with col1:
            try:
                img = Image.open(full_path)
                st.image(img, caption=of, use_container_width=True)
            except:
                st.warning(f"Can't open {of}")
        with col2:
            # Show Caption
            st.markdown(f"**📝 Caption:** {captions.get(of, 'No caption')}")
            # Show Tags
            tags = detections.get(of, [])
            if tags:
                st.markdown("**🏷 Tags:** " + ", ".join(tags))
            else:
                st.markdown("**🏷 Tags:** None")
            # Download button
            with open(full_path, "rb") as file:
                st.download_button(
                    label="📥 Download",
                    data=file,
                    file_name=of,
                    mime="image/jpeg"
                )
        st.markdown("---")
