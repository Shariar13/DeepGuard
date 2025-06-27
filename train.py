#!/usr/bin/env python3
"""
DeepGuard zero-shot ensemble training script.
Run as:
    python train.py

Generates: deepguard_cosine_ensemble.npz
"""

import os, gc, json, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from open_clip import create_model_and_transforms

# === CONFIG ===
dataset_dir = "Dataset"
output_file = "deepguard_cosine_ensemble.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_sizes = [224, 384, 512]

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
def embed_folder(backbone, prep_tf, folder, sizes, device):
    embs = []
    for f in tqdm(os.listdir(folder), desc=f"→ {os.path.basename(folder)}"):
        path = os.path.join(folder, f)
        try:
            img = Image.open(path).convert("RGB")
        except:
            continue
        vecs = []
        for sz in sizes:
            tf = prep_tf(sz)
            for p in patches(img.resize((sz, sz))):
                v = backbone(tf(p).unsqueeze(0).to(device)).squeeze(0)
                v = v / v.norm()
                vecs.append(v.cpu())
        if vecs:
            embs.append(torch.stack(vecs).mean(0).numpy())
    return np.vstack(embs)

# === MAIN EXECUTION ===
def main():
    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")

    # 1. CLIP ViT-H/14
    clip_model, _, _ = create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)
    clip_model.eval().requires_grad_(False)
    clip_embed = lambda tf: (lambda x: clip_model.encode_image(x))
    tf_clip = lambda s: build_tf(s, clip_norm)
    real_clip = embed_folder(clip_embed(tf_clip), tf_clip, real_dir, resize_sizes, device)
    fake_clip = embed_folder(clip_embed(tf_clip), tf_clip, fake_dir, resize_sizes, device)
    del clip_model; torch.cuda.empty_cache(); gc.collect()

    # 2. ResNet-50
    resnet = models.resnet50(weights="IMAGENET1K_V2").to(device)
    resnet.fc = torch.nn.Identity(); resnet.eval()
    tf_im = lambda s: build_tf(s, imagenet_norm)
    real_res = embed_folder(resnet, tf_im, real_dir, resize_sizes, device)
    fake_res = embed_folder(resnet, tf_im, fake_dir, resize_sizes, device)
    del resnet; torch.cuda.empty_cache(); gc.collect()

    # 3. EfficientNet-B0
    eff = models.efficientnet_b0(weights="IMAGENET1K_V1").to(device)
    eff.classifier = torch.nn.Identity(); eff.eval()
    real_eff = embed_folder(eff, tf_im, real_dir, resize_sizes, device)
    fake_eff = embed_folder(eff, tf_im, fake_dir, resize_sizes, device)
    del eff; torch.cuda.empty_cache(); gc.collect()

    # === SAVE NPZ ===
    np.savez(output_file,
             real_mean_clip=real_clip.mean(0), fake_mean_clip=fake_clip.mean(0),
             real_mean_res=real_res.mean(0), fake_mean_res=fake_res.mean(0),
             real_mean_eff=real_eff.mean(0), fake_mean_eff=fake_eff.mean(0),
             real_clip=real_clip, fake_clip=fake_clip,
             real_res=real_res, fake_res=fake_res,
             real_eff=real_eff, fake_eff=fake_eff)

    print(f"\n✅ Saved: {output_file}")
    meta = {
        "real": len(real_clip), "fake": len(fake_clip),
        "dims": {
            "clip": real_clip.shape[1],
            "res": real_res.shape[1],
            "eff": real_eff.shape[1]
        },
        "patches": 4, "sizes": resize_sizes
    }
    with open(output_file.replace(".npz", ".json"), "w") as jf:
        json.dump(meta, jf, indent=2)
    print("📄 Metadata:", meta)

if __name__ == "__main__":
    main()
