1mHiXKLMt5-MuPxr4HAJzLcQsFFS4qLAr
import os
import sys
import numpy as np
from PIL import Image
import torch
import open_clip
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


class Config:
    MEAN_PATH = "clip_means_v1.npz"
    CLIP_MODEL = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"
    SIZES = [224, 256, 288]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEST_REAL = "test/real"
    TEST_FAKE = "test/fake"


def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        Config.CLIP_MODEL, pretrained=Config.CLIP_PRETRAINED
    )
    model = model.to(Config.DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess


def extract_patches(image, size):
    image = image.resize((size, size))
    patch_size = size // 2
    patches = []
    for i in range(2):
        for j in range(2):
            box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
            patches.append(image.crop(box))
    return patches


def embed_image(img, model, preprocess):
    embeddings = []
    for size in Config.SIZES:
        patches = extract_patches(img, size)
        for patch in patches:
            tensor = preprocess(patch).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                emb = model.encode_image(tensor)
                emb = torch.nn.functional.normalize(emb, dim=-1)
                embeddings.append(emb.squeeze(0).cpu().numpy())
    return np.vstack(embeddings)


def load_class_means():
    if not os.path.exists(Config.MEAN_PATH):
        raise FileNotFoundError("Missing mean vectors. Run training script first.")
    data = np.load(Config.MEAN_PATH)
    return data["real"], data["fake"]


def predict_image(img_path, model, preprocess, real_mean, fake_mean):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open '{img_path}': {e}")
    emb = embed_image(img, model, preprocess)
    avg_emb = emb.mean(axis=0)
    real_score = float(np.dot(avg_emb, real_mean))
    fake_score = float(np.dot(avg_emb, fake_mean))
    label = 0 if real_score > fake_score else 1
    return label


def main():
    model, preprocess = load_clip()
    real_mean, fake_mean = load_class_means()

    y_true, y_pred = [], []
    
    for label, folder in [(0, Config.TEST_REAL), (1, Config.TEST_FAKE)]:
        for fname in tqdm(os.listdir(folder), desc=f"Evaluating {folder}"):
            fpath = os.path.join(folder, fname)
            try:
                pred = predict_image(fpath, model, preprocess, real_mean, fake_mean)
                y_true.append(label)
                y_pred.append(pred)
            except Exception as e:
                print(f"[WARN] Failed on {fname}: {e}", file=sys.stderr)

    print("\n✅ Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
    print("🧾 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
