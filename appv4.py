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
def load_faces():
    if os.path.exists(face_file):
        with open(face_file) as f:
            return json.load(f)
    return {}

model, preprocess = load_model()
vectors = load_embeddings()
faces = load_faces()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Smart Search Options")
    search_query = st.text_input("Search (e.g., 'man next to toilet')").strip()
    face_options = sorted(set(face for group in faces.values() for face in group))
    selected_face = st.selectbox("Filter by face cluster", [""] + face_options)
    uploaded = st.file_uploader("Upload a photo for reverse image search", type=["jpg", "jpeg", "png"])

# --- FILE PREP ---
all_files = list(vectors.keys())

# --- CLIP TEXT SEARCH ---
if search_query:
    with torch.no_grad():
        text = clip.tokenize([search_query]).to(model[0].device)
        text_feat = model[0].encode_text(text)[0].cpu().numpy()
    def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ranked = sorted([(cosine(text_feat, np.array(v)), k) for k, v in vectors.items()], reverse=True)
    all_files = [fname for _, fname in ranked[:48]]

# --- IMAGE SEARCH ---
elif uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Uploaded Image", use_container_width=True)

    with torch.no_grad():
        q_tensor = preprocess(query_img).unsqueeze(0).to(model[0].device)
        q_vec = model[0].encode_image(q_tensor)[0].cpu().numpy()
    def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ranked = sorted([(cosine(q_vec, np.array(v)), k) for k, v in vectors.items()], reverse=True)
    all_files = [fname for _, fname in ranked[:48]]

# --- FILTER BY FACE CLUSTER ---
if selected_face:
    all_files = [f for f in all_files if selected_face in faces.get(f, [])]

# --- DISPLAY IMAGES ---
selected_files = []
cols = st.columns(4)
for i, fname in enumerate(all_files):
    path = os.path.join(photo_dir, fname)
    if os.path.exists(path):
        img = Image.open(path)
        img.thumbnail((400, 400))
        col = cols[i % 4]
        col.image(img, caption=fname, use_container_width=True)
        if col.checkbox(f"Select {fname}", key=fname):
            selected_files.append(fname)

# --- DOWNLOAD ZIP ---
if selected_files:
    with io.BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for fname in selected_files:
                zip_file.write(os.path.join(photo_dir, fname), arcname=fname)
        st.download_button(
            label=f"📦 Download {len(selected_files)} selected",
            data=zip_buffer.getvalue(),
            file_name="selected_photos.zip",
            mime="application/zip"
        )
