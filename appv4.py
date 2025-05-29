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

# --- LOAD ---
captions = json.load(open(caption_file)) if os.path.exists(caption_file) else {}
detections = json.load(open(detection_file)) if os.path.exists(detection_file) else {}
clip_vectors = json.load(open(clip_file)) if os.path.exists(clip_file) else {}
face_labels = json.load(open(label_file)) if os.path.exists(label_file) else {}

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# --- UI SETUP ---
st.set_page_config(page_title="AI Photo Browser", layout="wide")
st.title("AI Photo Browser")
st.caption("Browse by face, search by text or image, and view AI-generated tags.")

# --- Search Bar at Top ---
query = st.text_input("Search your archive with natural language", placeholder="e.g. a child with helmet, a person riding a horse")

if query and clip_vectors:
    st.caption("Searching with CLIP...")
    with torch.no_grad():
        tokens = clip.tokenize([query]).to(device)
        text_vec = clip_model.encode_text(tokens)[0].cpu().numpy()

    def cosine(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    results = [(cosine(text_vec, np.array(v)), k) for k, v in clip_vectors.items()]
    top = sorted(results, reverse=True)[:12]

    st.subheader("Top Matches")
    cols = st.columns(4)
    for i, (score, fname) in enumerate(top):
        path = os.path.join(photo_root, fname)
        if os.path.exists(path):
            col = cols[i % 4]
            try:
                img = Image.open(path)
                caption = captions.get(fname, "-")
                col.image(img, caption=f"{fname}\nCaption: {caption}\nScore: {score:.2f}", use_container_width=True)
            except:
                col.warning(f"Error showing: {fname}")

# --- FACE GROUP SELECT ---
groups = sorted([g for g in os.listdir(face_root) if os.path.isdir(os.path.join(face_root, g))])
named_groups = [face_labels.get(g, g) for g in groups]
selected_group = st.sidebar.selectbox("Select a Person Group", named_groups)
selected_folder = groups[named_groups.index(selected_group)]

# --- FACE THUMBNAILS ---
st.subheader(f"Faces in {selected_group}")
thumbs = sorted(os.listdir(os.path.join(face_root, selected_folder)))
cols = st.columns(5)
for i, fname in enumerate(thumbs):
    col = cols[i % 5]
    try:
        img = Image.open(os.path.join(face_root, selected_folder, fname))
        col.image(img, caption=fname, use_container_width=True)
    except:
        col.warning(f"Couldn't load: {fname}")

# --- ORIGINAL PHOTOS FOR PERSON ---
st.markdown("---")
if st.checkbox("Show original photos containing this person"):
    st.subheader("Original Photo Matches")
    orig_files = sorted(set([f.split("_face_")[0] + ".jpg" for f in thumbs]))

    for i, of in enumerate(orig_files):
        path = os.path.join(photo_root, of)
        if not os.path.exists(path):
            continue

        col1, col2 = st.columns([1, 3])
        with col1:
            try:
                img = Image.open(path)
                st.image(img, caption=of, use_container_width=True)
            except:
                st.warning(f"Failed: {of}")
        with col2:
            st.markdown(f"**Caption:** {captions.get(of, 'None')}")
            tags = detections.get(of, [])
            st.markdown("**Tags:** " + (", ".join(tags) if tags else "None"))
            with open(path, "rb") as f:
                st.download_button("Download", f, file_name=of, mime="image/jpeg")
        st.markdown("---")

# --- REVERSE IMAGE SEARCH ---
st.markdown("---")
st.header("Reverse Image Search")

uploaded = st.file_uploader("Upload an image to find visually similar ones", type=["jpg", "jpeg", "png"])

if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Uploaded Image", use_container_width=True)

    with torch.no_grad():
        query_tensor = clip_preprocess(query_img).unsqueeze(0).to(device)
        query_vec = clip_model.encode_image(query_tensor)[0].cpu().numpy()

    results = [(cosine(query_vec, np.array(v)), k) for k, v in clip_vectors.items()]
    top = sorted(results, reverse=True)[:12]

    st.subheader("Top Matches:")
    cols = st.columns(4)
    for i, (score, fname) in enumerate(top):
        path = os.path.join(photo_root, fname)
        if os.path.exists(path):
            col = cols[i % 4]
            img = Image.open(path)
            caption = captions.get(fname, "-")
            col.image(img, caption=f"{fname}\nCaption: {caption}\nScore: {score:.2f}", use_container_width=True)
