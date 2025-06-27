
#!/usr/bin/env python3
"""
Test DeepGuard using cosine similarity and soft voting.
Folder structure:
  test/
    real/
    fake/
Usage:
  python test.py
"""

import os, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from open_clip import create_model_and_transforms
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
npz_file = "deepguard_cosine_ensemble.npz"
test_root = "test"
resize_sizes = [224, 384, 512]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD EMBEDDINGS ===
data = np.load(npz_file)
means = {
    "clip": (data['real_mean_clip'], data['fake_mean_clip']),
    "resnet": (data['real_mean_res'], data['fake_mean_res']),
    "eff": (data['real_mean_eff'], data['fake_mean_eff']),
}

# === NORMALIZERS ===
clip_norm = transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
imagenet_norm = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

def build_tf(size, norm):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        norm
    ])

def patches(img):
    w, h = img.size
    return [
        img.crop((0, 0, w//2, h//2)), img.crop((w//2, 0, w, h//2)),
        img.crop((0, h//2, w//2, h)), img.crop((w//2, h//2, w, h))
    ]

@torch.no_grad()
def embed_image(img, clip_model, resnet, eff):
    vecs = {"clip": [], "resnet": [], "eff": []}
    for sz in resize_sizes:
        # transforms
        tf_clip = build_tf(sz, clip_norm)
        tf_im = build_tf(sz, imagenet_norm)

        for patch in patches(img.resize((sz, sz))):
            p_clip = tf_clip(patch).unsqueeze(0).to(device)
            p_im = tf_im(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                v_clip = clip_model.encode_image(p_clip).squeeze(0)
                v_res = resnet(p_im).squeeze(0)
                v_eff = eff(p_im).squeeze(0)

            vecs["clip"].append((v_clip / v_clip.norm()).cpu().numpy())
            vecs["resnet"].append((v_res / v_res.norm()).cpu().numpy())
            vecs["eff"].append((v_eff / v_eff.norm()).cpu().numpy())

    return {k: np.mean(v, axis=0) for k,v in vecs.items() if v}

def predict(img_path, models):
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        return None

    emb = embed_image(img, *models)
    votes = []
    for k in means:
        real_vec, fake_vec = means[k]
        sim_real = cosine_similarity([emb[k]], [real_vec])[0][0]
        sim_fake = cosine_similarity([emb[k]], [fake_vec])[0][0]
        votes.append(sim_real > sim_fake)
    return int(sum(votes) >= 2)  # soft voting

# === LOAD MODELS ===
clip_model, _, _ = create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)
clip_model.eval().requires_grad_(False)

resnet = models.resnet50(weights="IMAGENET1K_V2").to(device)
resnet.fc = torch.nn.Identity(); resnet.eval()

eff = models.efficientnet_b0(weights="IMAGENET1K_V1").to(device)
eff.classifier = torch.nn.Identity(); eff.eval()

# === TEST LOOP ===
correct, total = 0, 0
for label in ["real", "fake"]:
    folder = os.path.join(test_root, label)
    y_true = 0 if label == "real" else 1
    for fname in tqdm(os.listdir(folder), desc=f"Testing {label}"):
        path = os.path.join(folder, fname)
        pred = predict(path, (clip_model, resnet, eff))
        if pred is None:
            continue
        total += 1
        correct += int(pred == y_true)

print(f"\n✅ Accuracy: {correct}/{total} = {correct/total:.2%}")
