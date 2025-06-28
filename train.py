# clip_distance_classifier.py — CLIP Embedding Distance-Based Real vs Fake Classifier (Commercial-Safe, No Filtering)

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip

# ------------------ Configuration ------------------ #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "dataset"
REAL_CLASS = "Real"
MEAN_VECTOR_PATH = "clip_mean_vectors.npz"
IMG_LIMIT_PER_CLASS = 200
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# ------------------ Compute Mean Vectors ------------------ #
def compute_class_means():
    print("🚀 Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.to(DEVICE).eval()

    real_feats, fake_feats = [], []
    print("📊 Extracting embeddings...")

    with torch.no_grad():
        for split in ["train", "val"]:
            split_dir = os.path.join(DATA_DIR, split)
            for cls_name in sorted(os.listdir(split_dir)):
                cls_path = os.path.join(split_dir, cls_name)
                label = 1 if cls_name == REAL_CLASS else 0
                count = 0

                for fname in tqdm(os.listdir(cls_path), desc=f"{split}/{cls_name}"):
                    if count >= IMG_LIMIT_PER_CLASS:
                        break
                    try:
                        img = Image.open(os.path.join(cls_path, fname)).convert("RGB")
                        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                        feat = model.encode_image(img_tensor)
                        feat /= feat.norm(dim=-1, keepdim=True)

                        if label == 1:
                            real_feats.append(feat.cpu().numpy())
                        else:
                            fake_feats.append(feat.cpu().numpy())
                        count += 1
                    except Exception:
                        continue

    if not real_feats or not fake_feats:
        print(f"❌ ERROR: Empty feature list — Real: {len(real_feats)}, Fake: {len(fake_feats)}")
        return

    print(f"📦 Final Feature Counts — Real: {len(real_feats)}, Fake: {len(fake_feats)}")
    real_mean = np.mean(np.vstack(real_feats), axis=0)
    fake_mean = np.mean(np.vstack(fake_feats), axis=0)
    np.savez(MEAN_VECTOR_PATH, real=real_mean, fake=fake_mean)
    print(f"✅ Mean vectors saved to: {MEAN_VECTOR_PATH}")

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
    print(f"🔍 Cosine Similarity - Real: {sim_real:.4f}, Fake: {sim_fake:.4f}")
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
