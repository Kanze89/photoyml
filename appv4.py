import streamlit as st
import os
import json
import clip
import torch
import numpy as np
from PIL import Image

# --- CONFIG ---
photo_dir = "../photoprism/originals/flickr30k_images"
embedding_file = "../clip_embeddings.json"
tag_file = "scene_combined.json"  # or detections_yolo.json if merged

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

model, preprocess = load_model()
vectors = load_embeddings()
tags = load_tags()

# --- SIDEBAR FILTER ---
st.sidebar.header("🔎 Tag Search")
search_tag = st.sidebar.text_input("Search by tag (e.g. horse, mountain, forest):").lower()

# --- UPLOAD IMAGE FOR REVERSE SEARCH ---
uploaded = st.file_uploader("Upload a photo to find similar images", type=["jpg", "jpeg", "png"])

# --- FILTER IMAGES BY TAG IF PROVIDED ---
all_files = list(vectors.keys())
if search_tag:
    all_files = [f for f in all_files if search_tag in [t.lower() for t in tags.get(f, [])]]

# --- PAGE CONTROLS ---
PAGE_SIZE = 12
page = st.sidebar.number_input("Page", 0, max(0, len(all_files) // PAGE_SIZE), step=1)
subset = all_files[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

# --- REVERSE IMAGE SEARCH ---
if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Uploaded Image", use_container_width=True)

    with torch.no_grad():
        query_tensor = preprocess(query_img).unsqueeze(0).to(model[0].device)
        query_vec = model[0].encode_image(query_tensor)[0].cpu().numpy()

    def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    results = [(cosine(query_vec, np.array(v)), k) for k, v in vectors.items()]
    results = sorted(results, reverse=True)[:PAGE_SIZE]
    subset = [fname for _, fname in results]

# --- DISPLAY IMAGES ---
cols = st.columns(4)
for i, fname in enumerate(subset):
    img_path = os.path.join(photo_dir, fname)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        img.thumbnail((400, 400))
        col = cols[i % 4]
        col.image(img, caption=fname, use_container_width=True)
        with open(img_path, "rb") as file:
            col.download_button("📥 Download", file.read(), file_name=fname)

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Fast & flexible AI photo archive — powered by CLIP, YOLO, and Places365.")
