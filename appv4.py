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

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.header("🔎 Filters")

    search_query = st.text_input("Search by text (CLIP)").strip().lower()
    tag_filter = st.text_input("Search by tag").strip().lower()

    show_faces = st.checkbox("Show face cluster filter", value=False)
    selected_face = ""
    if show_faces and faces:
        face_options = sorted(set(face for face_list in faces.values() for face in face_list))
        selected_face = st.selectbox("Filter by face cluster", [""] + face_options)

    uploaded = st.file_uploader("Upload image for reverse search", type=["jpg", "jpeg", "png"])

# --- IMAGE FILTERING ---
filtered_files = list(vectors.keys())

if tag_filter:
    filtered_files = [
        f for f in filtered_files
        if tag_filter in [t.lower() for t in tags.get(f, [])]
    ]

if selected_face:
    filtered_files = [
        f for f in filtered_files
        if selected_face in faces.get(f, [])
    ]

# --- CLIP TEXT SEARCH ---
if search_query:
    with torch.no_grad():
        text = clip.tokenize([search_query]).to(model[1])
        text_features = model[0].encode_text(text)[0].cpu().numpy()
    
    def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    scored = [(cosine(text_features, np.array(vectors[f])), f) for f in filtered_files]
    filtered_files = [f for _, f in sorted(scored, reverse=True)[:48]]

# --- REVERSE IMAGE SEARCH ---
if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Uploaded Image", use_container_width=True)

    with torch.no_grad():
        img_tensor = preprocess(query_img).unsqueeze(0).to(model[1])
        img_vec = model[0].encode_image(img_tensor)[0].cpu().numpy()

    def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    results = [(cosine(img_vec, np.array(v)), k) for k, v in vectors.items()]
    results = sorted(results, reverse=True)[:24]
    filtered_files = [fname for _, fname in results]

# --- DISPLAY IMAGES ---
selected_files = []
cols = st.columns(4)

for i, fname in enumerate(filtered_files[:48]):
    img_path = os.path.join(photo_dir, fname)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        img.thumbnail((400, 400))
        col = cols[i % 4]
        col.image(img, caption=fname, use_container_width=True)
        if col.checkbox(f"Select {fname}", key=fname):
            selected_files.append(fname)

# --- DOWNLOAD SELECTED ---
if selected_files:
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w") as zip_file:
            for fname in selected_files:
                zip_file.write(os.path.join(photo_dir, fname), arcname=fname)
        st.download_button(
            label=f"📦 Download {len(selected_files)} as ZIP",
            data=buffer.getvalue(),
            file_name="selected_photos.zip",
            mime="application/zip"
        )
