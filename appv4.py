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
photo_dir = os.path.abspath("../photoprism/originals/flickr30k_images")
embedding_file = os.path.abspath("../clip_embeddings.json")
tag_file = os.path.abspath("scene_combined.json")
face_file = os.path.abspath("face_clusters.json")

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Photo Explorer", layout="wide")
st.title("AI Photo Explorer")

# --- CACHING ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

@st.cache_data
def load_json_data(filepath):
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load {filepath}: {e}")
        return {}

model, preprocess, device = load_model()
vectors = load_json_data(embedding_file)
tags = load_json_data(tag_file)
faces = load_json_data(face_file)

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.header("Filters")
    search_query = st.text_input("CLIP Text Search").strip()
    search_tag = st.text_input("Search by tag").lower()
    face_options = sorted(set(face for faceset in faces.values() for face in faceset))
    selected_face = st.selectbox("Filter by face cluster", [""] + face_options)
    uploaded = st.file_uploader("Upload image for reverse search", type=["jpg", "jpeg", "png"])

# --- FILTERING ---
filtered_files = list(vectors.keys())

# Apply tag filter
if search_tag:
    filtered_files = [f for f in filtered_files if search_tag in [t.lower() for t in tags.get(f, [])]]

# Apply face cluster filter
if selected_face:
    filtered_files = [f for f in filtered_files if selected_face in faces.get(f, [])]

# --- CLIP TEXT SEARCH ---
if search_query:
    with torch.no_grad():
        try:
            text = clip.tokenize([search_query]).to(device)
            text_vec = model.encode_text(text)[0].cpu().numpy()
            def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            results = [(cosine(text_vec, np.array(v)), k) for k, v in vectors.items()]
            results = sorted(results, reverse=True)
            filtered_files = [fname for _, fname in results]
        except Exception as e:
            st.error(f"Text search failed: {e}")

# --- REVERSE IMAGE SEARCH ---
if uploaded:
    try:
        query_img = Image.open(uploaded).convert("RGB")
        st.image(query_img, caption="Uploaded Image", use_container_width=True)
        with torch.no_grad():
            q_tensor = preprocess(query_img).unsqueeze(0).to(device)
            q_vec = model.encode_image(q_tensor)[0].cpu().numpy()
        def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        results = [(cosine(q_vec, np.array(v)), k) for k, v in vectors.items()]
        results = sorted(results, reverse=True)
        filtered_files = [fname for _, fname in results]
    except Exception as e:
        st.error(f"Image search failed: {e}")

# --- DISPLAY IMAGES ---
selected_files = []
cols = st.columns(4)
for i, fname in enumerate(filtered_files[:48]):
    img_path = os.path.join(photo_dir, fname)
    try:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img.thumbnail((400, 400))
            col = cols[i % 4]
            col.image(img, caption=fname, use_container_width=True)
            if col.checkbox(f"Select {fname}", key=fname):
                selected_files.append(fname)
    except Exception as e:
        st.warning(f"Failed to load {fname}: {e}")

# --- DOWNLOAD SELECTED ---
if selected_files:
    try:
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
    except Exception as e:
        st.error(f"Failed to create ZIP file: {e}")
