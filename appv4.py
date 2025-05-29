import streamlit as st
import os
import json
import torch
import clip
import numpy as np
from PIL import Image

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
st.caption("Search by text or image. View original photos with captions and tags.")

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

# --- ORIGINAL PHOTOS (BASED ON A SINGLE FACE GROUP) ---
st.markdown("---")
st.subheader("Original Photos (Grouped)")

# Change this to the group folder you want to always show
group_folder = "Person_0"
thumbs_path = os.path.join(face_root, group_folder)

# Fallback in case folder doesn't exist
if not os.path.exists(thumbs_path):
    st.warning(f"Face group folder not found: {thumbs_path}")
    st.stop()

thumbs = sorted(os.listdir(thumbs_path))
orig_files = sorted(set([f.split("_face_")[0] + ".jpg" for f in thumbs]))

for i, of in enumerate(orig_files):
    path = os.path.join(photo_root, of)
    if not os.path.exists(path):
        continue

    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            img = Image.open(path)
            st.image(img, caption=of, use_container_width=True)
        except:
            st.warning(f"Failed to load: {of}")
    with col2:
        st.markdown(f"**Caption:** {captions.get(of, 'None')}")
        tags = detections.get(of, [])
        st.markdown("**Tags:** " + (", ".join(tags) if tags else "None"))
        with open(path, "rb") as f:
            st.download_button("Download Photo", f, file_name=of, mime="image/jpeg")
    st.markdown("---")
