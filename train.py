#!/usr/bin/env python3
"""
DeepGuard zero-shot generator.
Run as:
    python train.py
Generates: deepguard_clip_only.npz
"""

import os, gc, json, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from open_clip import create_model_and_transforms

# === CONFIG ===
dataset_dir = "Dataset"
output_file = "deepguard_clip_only.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_sizes = [224, 384, 512]

# === NORMALIZER ===
clip_norm = transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])

def build_tf(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        clip_norm
    ])

def patches(img):
    w, h = img.size
    return [
        img.crop((0, 0, w//2, h//2)), img.crop((w//2, 0, w, h//2)),
        img.crop((0, h//2, w//2, h)), img.crop((w//2, h//2, w, h))
    ]

@torch.no_grad()
def embed_folder(model, folder):
    embs = []
    for f in tqdm(os.listdir(folder), desc=f"→ {os.path.basename(folder)}"):
        path = os.path.join(folder, f)
        try:
            img = Image.open(path).convert("RGB")
        except:
            continue
        vecs = []
        for sz in resize_sizes:
            tf = build_tf(sz)
            for p in patches(img.resize((sz, sz))):
                v = model.encode_image(tf(p).unsqueeze(0).to(device)).squeeze(0)
                v = v / v.norm()
                vecs.append(v.cpu())
        if vecs:
            embs.append(torch.stack(vecs).mean(0).numpy())
    return np.vstack(embs)

# === MAIN EXECUTION ===
def main():
    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")

    # CLIP ViT-H/14
    model, _, _ = create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)
    model.eval().requires_grad_(False)

    real_clip = embed_folder(model, real_dir)
    fake_clip = embed_folder(model, fake_dir)
    del model; torch.cuda.empty_cache(); gc.collect()

    # SAVE NPZ
    np.savez(output_file,
             real_mean=real_clip.mean(0),
             fake_mean=fake_clip.mean(0),
             real=real_clip,
             fake=fake_clip)

    print(f"\n✅ Saved: {output_file}")
    meta = {
        "real": len(real_clip),
        "fake": len(fake_clip),
        "dim": real_clip.shape[1],
        "patches": 4,
        "sizes": resize_sizes
    }
    with open(output_file.replace(".npz", ".json"), "w") as jf:
        json.dump(meta, jf, indent=2)
    print("📄 Metadata:", meta)

if __name__ == "__main__":
    main()
