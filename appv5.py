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
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

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

model, preprocess, device = load_model()
vectors = load_embeddings()
tags = load_tags()
faces = load_faces()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Search & Filters")

    search_query = st.text_input("Search by tag or keyword").strip().lower()
    uploaded = st.file_uploader("Upload image for reverse search", type=["jpg", "jpeg", "png"])

    toggle_face = st.checkbox("Show face cluster filter")
    selected_face = None
    if toggle_face and faces:
        face_options = sorted({f for face_list in faces.values() for f in face_list})
        selected_face = st.selectbox("Face cluster", [""] + face_options)
        if selected_face == "":
            selected_face = None

# --- START IMAGE LIST ---
all_files = list(vectors.keys())

# --- FILTER BY TEXT QUERY ---
if search_query:
    text = clip.tokenize([search_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).cpu().numpy()[0]
    def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    scores = [(cosine(text_features, np.array(v)), k) for k, v in vectors.items()]
    scores = sorted(scores, reverse=True)[:200]
    all_files = [k for _, k in scores]

# --- FILTER BY FACE CLUSTER ---
if selected_face:
    all_files = [f for f in all_files if selected_face in faces.get(f, [])]

# --- REVERSE IMAGE SEARCH ---
if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Uploaded Image", use_container_width=True)
    with torch.no_grad():
        query_tensor = preprocess(query_img).unsqueeze(0).to(device)
        query_vec = model.encode_image(query_tensor)[0].cpu().numpy()

    def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    results = [(cosine(query_vec, np.array(vectors[f])), f) for f in all_files if f in vectors]
    results = sorted(results, reverse=True)[:48]
    all_files = [f for _, f in results]
    
# --- MULTI UPLOAD & PROCESS ---
st.header("📤 Upload New Photos to Archive")

multi_files = st.file_uploader("Upload photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if multi_files:
    st.info(f"{len(multi_files)} photos selected. Processing may take time.")
    progress = st.progress(0)
    new_photos = []

    for i, file in enumerate(multi_files):
        try:
            # Save file
            save_path = os.path.join(photo_dir, file.name)
            with open(save_path, "wb") as f:
                f.write(file.read())
            new_photos.append(file.name)

            # --- Run AI here ---
            st.text(f"Processing {file.name} ...")

            # Load Image
            image = Image.open(save_path).convert("RGB")

            # -- CAPTION with BLIP (optional)
            # caption = run_blip_caption(image)  # You can implement this
            # captions[file.name] = caption

            # -- TAG with YOLO + Scene (optional)
            # tags[file.name] = run_yolo_scene_tagging(image)

            # -- FACE DETECTION (optional)
            # faces[file.name] = run_face_detection(image)

            progress.progress((i + 1) / len(multi_files))
        except Exception as e:
            st.error(f"Failed to process {file.name}: {e}")

    st.success("✅ Upload & basic AI processing done!")
    st.text("Don't forget to re-run tagging scripts or refresh the UI to see them.")

# --- DISPLAY IMAGES ---
selected_files = []
cols = st.columns(4)
for i, fname in enumerate(all_files[:48]):
    path = os.path.join(photo_dir, fname)
    if os.path.exists(path):
        img = Image.open(path)
        col = cols[i % 4]
        col.image(img, caption=fname, use_container_width=True)
        if col.checkbox(f"Select {fname}", key=f"chk_{fname}"):
            selected_files.append(fname)

# --- DOWNLOAD ZIP ---
if selected_files:
    with io.BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for fname in selected_files:
                zip_file.write(os.path.join(photo_dir, fname), arcname=fname)
        st.download_button(
            label=f"Download {len(selected_files)} selected photos as ZIP",
            data=zip_buffer.getvalue(),
            file_name="photos.zip",
            mime="application/zip"
        )
