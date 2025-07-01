# test.py — Evaluate RealGuard v3.0 on unseen data

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import cv2

from open_clip import create_model_and_transforms

# === Load Saved Model === #
realguard_model = joblib.load("realguard_model.pkl")
clf = realguard_model["classifier"]
scaler = realguard_model["scaler"]
model_name = realguard_model["model_name"]
pretrained = realguard_model["pretrained"]

# === Load CLIP Model === #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(DEVICE).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
])

# === Feature Extractors === #
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

# === Load Test Features === #
def load_test_features(test_dir):
    features, labels = [], []
    for label, class_dir in enumerate(["real", "fake"]):
        class_path = os.path.join(test_dir, class_dir)
        for fname in tqdm(os.listdir(class_path), desc=f"Testing {class_dir}"):
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

# === Run Evaluation === #
X_test, y_test = load_test_features("DeepGuardDB/test")
X_scaled = scaler.transform(X_test)
y_pred = clf.predict(X_scaled)

# === Metrics === #
print("\n✅ Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
