# clip_distance_classifier_openclip.py — CLIP Embedding Distance-Based Real vs Fake Classifier (OpenCLIP Version with OpenAI-style Preprocessing)

import os
import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# ------------------ Configuration ------------------ #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ""
TRAIN_DIR = os.path.join(DATA_DIR, "train")
REAL_CLASS = "Real"
MEAN_VECTOR_PATH = "clip_mean_vectors.npz"
FAKE_IMG_LIMIT_PER_CLASS = 200
REAL_IMG_LIMIT_TOTAL = 1000
MARGIN_THRESHOLD = 0.009

# ------------------ Preprocessing (match OpenAI CLIP) ------------------ #
clip_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

# ------------------ Compute Mean Vectors ------------------ #
def compute_class_means():
    print("🚀 Loading OpenCLIP model...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion400m_e32")
    model = model.to(DEVICE)
    model.eval()

    real_feats, fake_feats = [], []
    real_img_count = 0

    print("📊 Extracting image embeddings for mean computation...")
    with torch.no_grad():
        for cls_name in sorted(os.listdir(TRAIN_DIR)):
            cls_path = os.path.join(TRAIN_DIR, cls_name)
            is_real = cls_name == REAL_CLASS
            count = 0

            for fname in tqdm(os.listdir(cls_path), desc=f"{cls_name}"):
                if is_real and real_img_count >= REAL_IMG_LIMIT_TOTAL:
                    break
                if not is_real and count >= FAKE_IMG_LIMIT_PER_CLASS:
                    break

                try:
                    img = Image.open(os.path.join(cls_path, fname)).convert("RGB")
                    img_tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
                    feat = model.encode_image(img_tensor)
                    feat /= feat.norm(dim=-1, keepdim=True)
                    if is_real:
                        real_feats.append(feat.cpu().numpy())
                        real_img_count += 1
                    else:
                        fake_feats.append(feat.cpu().numpy())
                        count += 1
                except Exception:
                    continue

    real_mean = np.mean(np.vstack(real_feats), axis=0)
    fake_mean = np.mean(np.vstack(fake_feats), axis=0)
    np.savez(MEAN_VECTOR_PATH, real=real_mean, fake=fake_mean)
    print(f"✅ Mean vectors saved to: {MEAN_VECTOR_PATH}")

# ------------------ Predict Function ------------------ #
def predict_image(image_path):
    print(f"📷 Loading image: {image_path}")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion400m_e32")
    model = model.to(DEVICE)
    model.eval()

    means = np.load(MEAN_VECTOR_PATH)
    mean_real = torch.tensor(means['real'], dtype=torch.float32).to(DEVICE)
    mean_fake = torch.tensor(means['fake'], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
            img_feat = model.encode_image(img_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
        except Exception as e:
            print(f"❌ Failed to process image: {e}")
            return

    sim_real = torch.cosine_similarity(img_feat, mean_real.unsqueeze(0)).item()
    sim_fake = torch.cosine_similarity(img_feat, mean_fake.unsqueeze(0)).item()
    margin = abs(sim_real - sim_fake)

    if sim_real > sim_fake and margin < MARGIN_THRESHOLD:
        label = "FAKE"
    elif sim_fake > sim_real and margin < MARGIN_THRESHOLD:
        label = "REAL"
    else:
        label = "REAL" if sim_real > sim_fake else "FAKE"

    conf = max(sim_real, sim_fake) * 100
    print(f"🔍 Cosine Similarity - Real: {sim_real:.4f}, Fake: {sim_fake:.4f}, Margin: {margin:.4f}")
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
        print("   python clip_distance_classifier_openclip.py build          # to compute mean vectors")
        print("   python clip_distance_classifier_openclip.py <image_path>  # to predict image")
