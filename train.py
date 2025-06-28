# clip_distance_classifier.py — Commercial-Safe, Balanced CLIP Real vs Fake Classifier (OpenCLIP + Clean Filtering)

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip

# ------------------ Configuration ------------------ #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data"  # Updated dataset structure: data/real and data/fake
REAL_FOLDER = "real"
FAKE_FOLDER = "fake"
MEAN_VECTOR_PATH = "clip_mean_vectors.npz"
IMG_LIMIT_PER_CLASS = 200
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
SEED = 42

# ------------------ Seed Fix for Reproducibility ------------------ #
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------ Compute Clean Mean Vectors ------------------ #
def compute_class_means():
    print("🚀 Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.to(DEVICE).eval()

    all_feats = []
    all_labels = []

    print("📊 Step 1: Gathering all embeddings...")
    with torch.no_grad():
        for label_name, label_val, max_img in [(REAL_FOLDER, 1, IMG_LIMIT_PER_CLASS), (FAKE_FOLDER, 0, IMG_LIMIT_PER_CLASS)]:
            path = os.path.join(DATA_DIR, label_name)
            count = 0
            for fname in tqdm(os.listdir(path), desc=label_name):
                if count >= max_img:
                    break
                try:
                    img = Image.open(os.path.join(path, fname)).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                    feat = model.encode_image(img_tensor)
                    feat /= feat.norm(dim=-1, keepdim=True)
                    all_feats.append(feat.cpu().numpy())
                    all_labels.append(label_val)
                    count += 1
                except Exception:
                    continue

    all_feats = np.vstack(all_feats)
    all_labels = np.array(all_labels)

    print("🧹 Step 2: Clean filtering based on self-alignment...")
    temp_real_mean = np.mean(all_feats[all_labels == 1], axis=0)
    temp_fake_mean = np.mean(all_feats[all_labels == 0], axis=0)

    real_feats_clean = []
    fake_feats_clean = []

    for feat, label in zip(all_feats, all_labels):
        sim_real = np.dot(feat, temp_real_mean)
        sim_fake = np.dot(feat, temp_fake_mean)
        if label == 1 and sim_real > sim_fake:
            real_feats_clean.append(feat)
        elif label == 0 and sim_fake > sim_real:
            fake_feats_clean.append(feat)

    if not real_feats_clean or not fake_feats_clean:
        print(f"❌ ERROR: Clean features empty — Real: {len(real_feats_clean)}, Fake: {len(fake_feats_clean)}")
        return

    real_mean = np.mean(np.vstack(real_feats_clean), axis=0)
    fake_mean = np.mean(np.vstack(fake_feats_clean), axis=0)
    np.savez(MEAN_VECTOR_PATH, real=real_mean, fake=fake_mean)
    print(f"✅ Saved filtered mean vectors to: {MEAN_VECTOR_PATH}")

# ------------------ Predict Function ------------------ #
def predict_image(image_path):
    print(f"📷 Loading image: {image_path}")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.to(DEVICE).eval()

    means = np.load(MEAN_VECTOR_PATH)
    mean_real = torch.tensor(means['real'], dtype=torch.float32).to(DEVICE)
    mean_fake = torch.tensor(means['fake'], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            img_feat = model.encode_image(img_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
        except Exception as e:
            print(f"❌ Failed to process image: {e}")
            return

    sim_real = torch.cosine_similarity(img_feat, mean_real.unsqueeze(0)).item()
    sim_fake = torch.cosine_similarity(img_feat, mean_fake.unsqueeze(0)).item()

    label = "REAL" if sim_real > sim_fake else "FAKE"
    conf = max(sim_real, sim_fake) * 100
    print(f"🔍 Cosine Similarity — Real: {sim_real:.4f}, Fake: {sim_fake:.4f}")
    print(f"✅ Prediction: {label} ({conf:.2f}% confident)")

# ------------------ CLI ------------------ #
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "build":
        compute_class_means()
    elif len(sys.argv) == 2:
        predict_image(sys.argv[1])
    else:
        print("❌ Usage:")
        print("   python clip_distance_classifier.py build          # to compute mean vectors")
        print("   python clip_distance_classifier.py <image_path>  # to predict image")
