import streamlit as st
import os
import json
import torch
import clip
import numpy as np
from PIL import Image
import zipfile
import io

# --- CONFIG ---
face_root = "../face_clusters"
photo_root = "../photoprism/originals/flickr30k_images"
caption_file = "../captions_blip.json"
detection_file = "../detections_yolo.json"
clip_file = "../clip_embeddings.json"
label_file = "../face_labels.json"

# --- LOAD DATA ---
captions = json.load(open(caption_file)) if os.path.exists(caption_file) else {}
detections = json.load(open(detection_file)) if os.path.exists(detection_file) else {}
clip_vectors = json.load(open(clip_file)) if os.path.exists(clip_file) else {}
face_labels = json.load(open(label_file)) if os.path.exists(label_file) else {}

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Photo Browser", layout="wide")
st.title("AI Photo Browser")
st.caption("Search by text or image. Filter by tags and faces. Download results.")

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- TEXT SEARCH ---
query = st.text_input("Text Search", placeholder="e.g. a child wearing helmet")

if query and clip_vectors:
    st.caption("Searching with CLIP text encoder...")
    with torch.no_grad():
        tokens = clip.tokenize([query]).to(device)
        text_vec = clip_model.encode_text(tokens)[0].cpu().numpy()

    results = [(cosine(text_vec, np.array(v)), k) for k, v in clip_vectors.items()]
    top = sorted(results, reverse=True)[:12]

    st.subheader("Top Matches (Text Search)")
    cols = st.columns(4)
    for i, (score, fname) in enumerate(top):
        path = os.path.join(photo_root, fname)
        if os.path.exists(path):
            col = cols[i % 4]
            img = Image.open(path)
            caption = captions.get(fname, "—")
            col.image(img, caption=f"{fname}\n{caption}\nScore: {score:.2f}", use_container_width=True)

# --- REVERSE IMAGE SEARCH ---
st.markdown("---")
st.subheader("Reverse Image Search (Upload a Photo)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Uploaded Image", use_container_width=True)

    with torch.no_grad():
        query_tensor = clip_preprocess(query_img).unsqueeze(0).to(device)
        query_vec = clip_model.encode_image(query_tensor)[0].cpu().numpy()

    results = [(cosine(query_vec, np.array(v)), k) for k, v in clip_vectors.items()]
    top = sorted(results, reverse=True)[:12]

    st.subheader("Top Matches (Image Similarity)")
    cols = st.columns(4)
    for i, (score, fname) in enumerate(top):
        path = os.path.join(photo_root, fname)
        if os.path.exists(path):
            col = cols[i % 4]
            img = Image.open(path)
            caption = captions.get(fname, "—")
            col.image(img, caption=f"{fname}\n{caption}\nScore: {score:.2f}", use_container_width=True)

# --- TAG + FACE SEARCH COMBO ---
st.markdown("---")
st.subheader("📂 Filter by Person + Tags")

# Load face groups and tags
groups = sorted([g for g in os.listdir(face_root) if os.path.isdir(os.path.join(face_root, g))])
all_tags = sorted(set(tag for tags in detections.values() for tag in tags))
named_groups = [face_labels.get(g, g) for g in groups]

# Clear Filters
if "person_select" not in st.session_state:
    st.session_state.person_select = named_groups[0]
if "tag_select" not in st.session_state:
    st.session_state.tag_select = []

if st.button("🔄 Clear Filters"):
    st.session_state["person_select"] = named_groups[0]
    st.session_state["tag_select"] = []
    st.experimental_rerun()

# Select person and tags
selected_name = st.selectbox("Select Person", named_groups, key="person_select")
selected_group = groups[named_groups.index(selected_name)]
selected_tags = st.multiselect("Select Tags", all_tags, key="tag_select")

# Get face-matched photos
thumbs_path = os.path.join(face_root, selected_group)
thumbs = sorted(os.listdir(thumbs_path))
orig_files = sorted(set([f.split("_face_")[0] + ".jpg" for f in thumbs]))

def matches_tags(file):
    photo_tags = detections.get(file, [])
    return all(tag in photo_tags for tag in selected_tags)

filtered_files = [f for f in orig_files if matches_tags(f)]

# Display results
st.markdown("---")
st.subheader("Filtered Results")

selected_downloads = []

if not filtered_files:
    st.warning("No matching photos found.")
else:
    for of in filtered_files:
        path = os.path.join(photo_root, of)
        if not os.path.exists(path):
            continue

        col1, col2 = st.columns([1, 2])
        with col1:
            img = Image.open(path)
            st.image(img, caption=of, use_container_width=True)
        with col2:
            st.markdown(f"**Caption:** {captions.get(of, '—')}")
            tags = detections.get(of, [])
            st.markdown("**Tags:** " + (", ".join(tags) if tags else "None"))
            with open(path, "rb") as f:
                st.download_button("Download Photo", f, file_name=of, mime="image/jpeg")
            if st.checkbox(f"Include {of}", key=f"chk_{of}"):
                selected_downloads.append(path)
        st.markdown("---")

# Bulk ZIP download
if selected_downloads:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for file in selected_downloads:
            arcname = os.path.basename(file)
            zipf.write(file, arcname)
    zip_buffer.seek(0)
    st.download_button(
        label="⬇️ Download Selected as ZIP",
        data=zip_buffer,
        file_name="selected_photos.zip",
        mime="application/zip"
    )
