# RealGuard v3.0: Robust AI vs Real Image Detector
# Dataset: DeepGuardDB/train/real and DeepGuardDB/train/fake
# License-Free, Real-World Ready, Patch-Consistent, CLIP+Forensic Fusion

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from open_clip import create_model_and_transforms
import cv2
import joblib

# === SETUP === #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

model, preprocess, _ = create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(DEVICE).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
])

# === FEATURE EXTRACTORS === #
def extract_clip_embedding(img: Image.Image):
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(img_tensor).cpu().numpy().flatten()
    return embedding / np.linalg.norm(embedding)

def extract_forensic_features(image_np: np.ndarray):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    dct = cv2.dct(np.float32(gray) / 255.0)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.mean(np.abs(sobelx) + np.abs(sobely))
    return np.concatenate([
        dct[:8, :8].flatten(),
        hist / (hist.sum() + 1e-8),
        [edge]
    ])

def extract_patch_coherence_features(img: Image.Image):
    img = img.resize((224, 224))
    patches = []
    for i in range(3):
        for j in range(3):
            patch = img.crop((j*75, i*75, j*75+75, i*75+75))
            emb = extract_clip_embedding(patch)
            patches.append(emb)
    patch_embeddings = np.stack(patches)
    stddev = np.std(patch_embeddings, axis=0)
    return np.array([np.mean(stddev)])

# === DATASET LOADING === #
def load_features(data_dir):
    features, labels = [], []
    for label, class_dir in enumerate(["real", "fake"]):
        class_path = os.path.join(data_dir, class_dir)
        for fname in tqdm(os.listdir(class_path), desc=class_dir):
            try:
                img_path = os.path.join(class_path, fname)
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)

                clip_emb = extract_clip_embedding(img)
                forensic = extract_forensic_features(img_np)
                patch_std = extract_patch_coherence_features(img)

                feature_vec = np.concatenate([clip_emb, forensic, patch_std])
                features.append(feature_vec)
                labels.append(label)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    return np.array(features), np.array(labels)

# === MAIN PIPELINE === #
features, labels = load_features("DeepGuardDB/train")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_scaled, labels)

# === SAVE FINAL MODEL (UNIFIED FOR DEPLOYMENT) === #
realguard_model = {
    "classifier": clf,
    "scaler": scaler,
    "model_name": MODEL_NAME,
    "pretrained": PRETRAINED
}

joblib.dump(realguard_model, "realguard_model.pkl")

print("✅ RealGuard v3.0 trained and saved as unified .pkl model for deployment.")
