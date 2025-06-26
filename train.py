# ✅ Updated: Final Single-File Version of DeepGuard (For Patent Submission)
# Includes scale similarity fusion (no compression), and removes val folder usage

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import open_clip
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
class Config:
    TRAIN_REAL = "DeepGuardDB/train/real"
    TRAIN_FAKE = "DeepGuardDB/train/fake"

    MODEL_SAVE_PATH = "consistency_detector.pth"
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_DIM = 1024
    REAL_PENALTY = 4.0

# ==================== DATASET ====================
class SimpleDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        valid_extensions = (
            '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
            '.webp', '.svg', '.ico', '.jfif', '.pjpeg', '.pjp', '.avif',
            '.apng', '.heic', '.heif', '.raw', '.cr2', '.nef', '.arw',
            '.dng', '.orf', '.rw2', '.pef', '.sr2', '.x3f', '.raf',
            '.3fr', '.fff', '.iiq', '.k25', '.kdc', '.mef', '.mos',
            '.mrw', '.nrw', '.ptx', '.r3d', '.rwl', '.srw', '.erf',
            '.dcr', '.crw', '.bay', '.cap', '.eip', '.dcs',
            '.dcx', '.djvu', '.eps', '.exr', '.hdr', '.j2k', '.jp2',
            '.jpx', '.mng', '.pbm', '.pcx', '.pfm', '.pgm', '.ppm',
            '.psd', '.ras', '.sgi', '.tga', '.wbmp', '.xbm', '.xpm'
        )

        for img_name in os.listdir(real_dir):
            if img_name.lower().endswith(valid_extensions):
                self.images.append(os.path.join(real_dir, img_name))
                self.labels.append(0)

        for img_name in os.listdir(fake_dir):
            if img_name.lower().endswith(valid_extensions):
                self.images.append(os.path.join(fake_dir, img_name))
                self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            return torch.zeros(3, 224, 224), self.labels[idx]

# ==================== MODEL ====================
class ConsistencyDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.consistency_head = nn.Sequential(
            nn.Linear(Config.CLIP_DIM, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

        self.final_classifier = nn.Sequential(
            nn.Linear(Config.CLIP_DIM + 3, 128),  # +3 for 3 cosine similarities
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def forward(self, full, half, quarter):
        c1 = F.cosine_similarity(full, half, dim=1).unsqueeze(1)
        c2 = F.cosine_similarity(full, quarter, dim=1).unsqueeze(1)
        c3 = F.cosine_similarity(half, quarter, dim=1).unsqueeze(1)
        scale_features = torch.cat([c1, c2, c3], dim=1)

        traditional_logits = self.consistency_head(full)
        fusion_input = torch.cat([full, scale_features], dim=1)
        final_logits = self.final_classifier(fusion_input)

        return final_logits, scale_features, traditional_logits

# ==================== FEATURE EXTRACTION ====================
def extract_multiscale_features(batch, clip_model, clip_pre):
    all_f, all_h, all_q = [], [], []
    for img in batch:
        img = transforms.ToPILImage()(img)
        t_full = clip_pre(img.resize((224, 224))).unsqueeze(0).to(Config.DEVICE)
        t_half = clip_pre(img.resize((112, 112)).resize((224, 224))).unsqueeze(0).to(Config.DEVICE)
        t_quarter = clip_pre(img.resize((56, 56)).resize((224, 224))).unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            f = F.normalize(clip_model.encode_image(t_full), dim=-1).squeeze()
            h = F.normalize(clip_model.encode_image(t_half), dim=-1).squeeze()
            q = F.normalize(clip_model.encode_image(t_quarter), dim=-1).squeeze()

        all_f.append(f); all_h.append(h); all_q.append(q)
    return torch.stack(all_f), torch.stack(all_h), torch.stack(all_q)

# ==================== LOSS ====================
def consistency_loss(logits, scale_feats, trad_logits, labels):
    weights = torch.where(labels == 0, Config.REAL_PENALTY, 1.0).to(Config.DEVICE)
    main = F.cross_entropy(logits, labels, reduction='none')
    w_loss = (main * weights).mean()
    scale_target = (1 - labels.float()).unsqueeze(1).expand(-1, 3)
    scale_loss = F.mse_loss(scale_feats, scale_target)
    trad_loss = F.cross_entropy(trad_logits, labels)
    return w_loss + 0.1 * scale_loss + 0.05 * trad_loss

# ==================== TRAIN ====================
def train_epoch(model, loader, optim, epoch, clip_model, clip_pre):
    model.train(); total, correct, loss_sum = 0, 0, 0
    for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
        imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
        f, h, q = extract_multiscale_features(imgs, clip_model, clip_pre)
        logits, scale, trad = model(f, h, q)
        loss = consistency_loss(logits, scale, trad, labels)
        optim.zero_grad(); loss.backward(); optim.step()
        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item(); total += labels.size(0); loss_sum += loss.item()
    return loss_sum / len(loader), 100. * correct / total

# ==================== MAIN ====================
def main():
    print(f"🚀 DeepGuard Model | Device: {Config.DEVICE}")
    clip_model, _, clip_pre = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    clip_model = clip_model.to(Config.DEVICE).eval()
    for p in clip_model.parameters(): p.requires_grad = False

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    data = SimpleDataset(Config.TRAIN_REAL, Config.TRAIN_FAKE, tf)
    loader = DataLoader(data, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)

    model = ConsistencyDetector().to(Config.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        clip_model = nn.DataParallel(clip_model)

    optim = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    best = 0
    for e in range(Config.EPOCHS):
        loss, acc = train_epoch(model, loader, optim, e, clip_model, clip_pre)
        if acc > best:
            best = acc
            torch.save({'model_state_dict': model.state_dict(), 'accuracy': acc}, Config.MODEL_SAVE_PATH)
            print(f"✅ Saved best model @ {acc:.2f}%")

    print("\n🎉 Training Complete | Final Accuracy: {:.2f}%".format(best))

if __name__ == '__main__':
    main()
