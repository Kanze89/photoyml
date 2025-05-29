import streamlit as st
import os
import json
import clip
import torch
import numpy as np
from PIL import Image
import zipfile
import io

# --- CONFIG ---
photo_dir = "../photoprism/originals/flickr30k_images"
embedding_file = "../clip_embeddings.json"
tag_file = "scene_combined.json"
face_file = "face_clusters.json"

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Photo Explorer", layout="wide")
st.title("AI Photo Explorer")

# --- CACHING ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return clip.load("ViT-B/32", device=device)

@st.cache_data
def load_embeddings():
    with open(embedding_file) as f:
        return json.load(f)

@st.cache_data
def load_tags():
    if os.path.exists(tag_file):
        with open(tag_file) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_faces():
    if os.path.exists(face_file):
        with open(face_file) as f:
            return json.load(f)
    return {}

model, preprocess = load_model()
vectors = load_embeddings()
tags = load_tags()
faces = load_faces()

# --- FILTER BAR ---
with st.sidebar:
    st.header("🔎 Filters")
    search_tag = st.text_input("Search by tag").lower()
    face_options = sorted(set(face for faceset in faces.values() for face in faceset))
    selected_face = st.selectbox("Filter by face cluster", [""] + face_options)
    uploaded = st.file_uploader("Upload image for reverse search", type=["jpg", "jpeg", "png"])

# --- IMAGE FILTERING ---
all_files = list(vectors.keys())
if search_tag:
    all_files = [f for f in all_files if search_tag in [t.lower() for t in tags.get(f, [])]]
if selected_face:
    all_files = [f for f in all_files if selected_face in faces.get(f, [])]

# --- REVERSE SEARCH ---
if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Uploaded Image", use_container_width=True)

    with torch.no_grad():
        query_tensor = preprocess(query_img).unsqueeze(0).to(model[0].device)
        query_vec = model[0].encode_image(query_tensor)[0].cpu().numpy()

    def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    results = [(cosine(query_vec, np.array(v)), k) for k, v in vectors.items()]
    results = sorted(results, reverse=True)[:24]
    all_files = [fname for _, fname in results]

# --- DISPLAY IMAGES ---
selected_files = []
cols = st.columns(4)
for i, fname in enumerate(all_files[:48]):
    img_path = os.path.join(photo_dir, fname)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        img.thumbnail((400, 400))
        col = cols[i % 4]
        col.image(img, caption=fname, use_container_width=True)
        if col.checkbox(f"Select {fname}", key=fname):
            selected_files.append(fname)

# --- DOWNLOAD ALL SELECTED ---
if selected_files:
    with io.BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for fname in selected_files:
                fpath = os.path.join(photo_dir, fname)
                zip_file.write(fpath, arcname=fname)
        st.download_button(
            label=f"📦 Download {len(selected_files)} selected as ZIP",
            data=zip_buffer.getvalue(),
            file_name="selected_photos.zip",
            mime="application/zip"
        )
