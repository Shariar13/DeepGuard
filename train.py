import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import open_clip
from tqdm import tqdm

# ==================== CONFIG ====================
class Config:
    DATA_REAL = "DeepGuardDB/val/real"
    DATA_FAKE = "DeepGuardDB/val/fake"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SIZES = [224, 256, 288]
    PATCHES = 4
    CLIP_MODEL = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"
    THRESHOLD = 0.0  # Cosine distance threshold (auto-adjusted by class mean comparison)

# ==================== LOAD CLIP ====================
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    Config.CLIP_MODEL, pretrained=Config.CLIP_PRETRAINED
)
clip_model = clip_model.to(Config.DEVICE).eval()
for p in clip_model.parameters():
    p.requires_grad = False

# ==================== FEATURE EXTRACTION ====================
def extract_patches(image, size):
    image = image.resize((size, size))
    patch_size = size // 2
    patches = []
    for i in range(2):
        for j in range(2):
            box = (j * patch_size, i * patch_size, (j+1) * patch_size, (i+1) * patch_size)
            patches.append(image.crop(box))
    return patches

def embed_image(image):
    embeddings = []
    for size in Config.SIZES:
        patches = extract_patches(image, size)
        for patch in patches:
            tensor = clip_preprocess(patch).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                emb = clip_model.encode_image(tensor)
                emb = torch.nn.functional.normalize(emb, dim=-1)
                embeddings.append(emb.squeeze(0).cpu().numpy())
    return np.array(embeddings)

def build_class_mean(directory):
    embs = []
    for fname in tqdm(os.listdir(directory), desc=f"Embedding {directory}"):
        path = os.path.join(directory, fname)
        try:
            img = Image.open(path).convert("RGB")
            emb = embed_image(img)
            embs.append(emb.mean(axis=0))
        except:
            continue
    return np.mean(np.stack(embs), axis=0)

# ==================== MAIN ====================
def main():
    print("\U0001F50D Building class means...")
    real_mean = build_class_mean(Config.DATA_REAL)
    fake_mean = build_class_mean(Config.DATA_FAKE)

    def predict(image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            emb = embed_image(img)
            avg_emb = emb.mean(axis=0)
            real_dist = np.dot(avg_emb, real_mean)
            fake_dist = np.dot(avg_emb, fake_mean)
            return 0 if real_dist > fake_dist else 1  # 0=Real, 1=Fake
        except:
            return -1

    print("\U0001F4CA Validating...")
    y_true, y_pred = [], []
    for label, folder in [(0, Config.DATA_REAL), (1, Config.DATA_FAKE)]:
        for fname in tqdm(os.listdir(folder), desc=f"Predicting {folder}"):
            path = os.path.join(folder, fname)
            pred = predict(path)
            if pred != -1:
                y_true.append(label)
                y_pred.append(pred)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\u2705 Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
    print("\U0001F9FE Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
