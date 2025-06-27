import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms, models
from open_clip import create_model_and_transforms

# === CONFIG ===
dataset_dir = "Dataset"  # Must contain 'real/' and 'fake/' folders
output_npz = "deepguard_cosine_ensemble.npz"
device = "cuda" if torch.cuda.is_available() else "cpu"
resize_sizes = [224, 384, 512]

# === LOAD MODELS ===
clip_model, _, _ = create_model_and_transforms(
    model_name="ViT-H-14",
    pretrained="laion2b_s32b_b79k",
    device=device
)
clip_model.eval()

resnet = models.resnet50(pretrained=True).to(device)
resnet.fc = torch.nn.Identity()
resnet.eval()

efficientnet = models.efficientnet_b0(pretrained=True).to(device)
efficientnet.classifier = torch.nn.Identity()
efficientnet.eval()

# === TRANSFORMS ===
clip_norm = transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
imagenet_norm = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

def get_transform(size, norm_type):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        clip_norm if norm_type == 'clip' else imagenet_norm
    ])

def split_into_patches(img):
    w, h = img.size
    return [
        img.crop((0, 0, w//2, h//2)),
        img.crop((w//2, 0, w, h//2)),
        img.crop((0, h//2, w//2, h)),
        img.crop((w//2, h//2, w, h))
    ]

@torch.no_grad()
def compute_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    emb_clip, emb_resnet, emb_eff = [], [], []

    for size in resize_sizes:
        resized = img.resize((size, size))
        patches = split_into_patches(resized)
        for patch in patches:
            # CLIP
            t_clip = get_transform(size, 'clip')(patch).unsqueeze(0).to(device)
            e_clip = clip_model.encode_image(t_clip).squeeze(0)
            emb_clip.append(e_clip / e_clip.norm())

            # ResNet
            t_imagenet = get_transform(size, 'imagenet')(patch).unsqueeze(0).to(device)
            e_res = resnet(t_imagenet).squeeze(0)
            emb_resnet.append(e_res / e_res.norm())

            # EfficientNet
            e_eff = efficientnet(t_imagenet).squeeze(0)
            emb_eff.append(e_eff / e_eff.norm())

    return {
        "clip": torch.stack(emb_clip).mean(0).cpu().numpy(),
        "resnet": torch.stack(emb_resnet).mean(0).cpu().numpy(),
        "efficient": torch.stack(emb_eff).mean(0).cpu().numpy()
    }

def process_folder(path):
    clip_list, res_list, eff_list = [], [], []
    for fname in tqdm(os.listdir(path), desc=f"Processing {os.path.basename(path)}"):
        fpath = os.path.join(path, fname)
        try:
            emb = compute_embedding(fpath)
            clip_list.append(emb["clip"])
            res_list.append(emb["resnet"])
            eff_list.append(emb["efficient"])
        except Exception as e:
            print(f"⚠️ Skipped {fname}: {e}")
            continue
    return {
        "clip": np.array(clip_list),
        "resnet": np.array(res_list),
        "efficient": np.array(eff_list)
    }

if __name__ == "__main__":
    real_data = process_folder(os.path.join(dataset_dir, "real"))
    fake_data = process_folder(os.path.join(dataset_dir, "fake"))

    np.savez(output_npz,
             real_mean_clip=np.mean(real_data["clip"], axis=0),
             fake_mean_clip=np.mean(fake_data["clip"], axis=0),
             real_mean_resnet=np.mean(real_data["resnet"], axis=0),
             fake_mean_resnet=np.mean(fake_data["resnet"], axis=0),
             real_mean_efficient=np.mean(real_data["efficient"], axis=0),
             fake_mean_efficient=np.mean(fake_data["efficient"], axis=0),
             real_clip=real_data["clip"],
             fake_clip=fake_data["clip"],
             real_resnet=real_data["resnet"],
             fake_resnet=fake_data["resnet"],
             real_efficient=real_data["efficient"],
             fake_efficient=fake_data["efficient"])

    print(f"\n✅ .npz file saved: {output_npz}")
