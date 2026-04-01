"""
猫狗分类 - 基于 ConvNeXt V2 的 SOTA 方案
支持: 混合精度训练、MixUp/CutMix、TTA、训练曲线保存
"""

import os
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────── 配置 ────────────────────────────────
class CFG:
    # 路径
    data_dir   = Path("/data/smx/catordog/dataset")
    train_dir  = data_dir / "train"
    test_dir   = data_dir / "test"
    output_dir = Path("/data/smx/catordog/output")

    # 模型
    model_name  = "convnext_large_in22ft1k"
    pretrained  = True
    num_classes = 1          # 二分类用 sigmoid

    # 训练
    image_size  = 384
    batch_size  = 32
    num_epochs  = 15
    val_ratio   = 0.2        # 验证集比例

    # 优化器
    lr           = 2e-4
    weight_decay = 1e-2
    min_lr       = 1e-6
    warmup_epochs= 1

    # 正则化
    label_smoothing = 0.05
    mixup_alpha     = 0.4
    cutmix_alpha    = 1.0
    mix_prob        = 0.5

    # 推理 TTA (1=关闭, 3=推荐, 5=最强)
    tta_steps = 3

    # 其他
    seed        = 42
    num_workers = 12
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    amp         = True


CFG.output_dir.mkdir(parents=True, exist_ok=True)


# ────────────────────────── 工具函数 ─────────────────────────────────
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import log_loss
    return log_loss(y_true, y_pred)


# ─���───────────────────────── 数据集 ──────────────────────────────────
def build_dataframe() -> pd.DataFrame:
    """构建训练集 DataFrame，label: 0=cat 1=dog"""
    paths = list(CFG.train_dir.glob("*.jpg"))
    labels = [1 if p.stem.startswith("dog") else 0 for p in paths]
    df = pd.DataFrame({"path": paths, "label": labels})
    df = df.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    return df


def get_transforms(mode: str) -> A.Compose:
    size = CFG.image_size
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
            ], p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1, p=0.5),
            A.CoarseDropout(num_holes_range=(1, 8),
                            hole_height_range=(size//16, size//8),
                            hole_width_range=(size//16, size//8), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:  # valid / test
        return A.Compose([
            A.Resize(height=size, width=size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_tta_transforms() -> list:
    """TTA: 原图 + 水平翻转 + 多尺度裁剪"""
    size = CFG.image_size
    tfms = [
        A.Compose([A.Resize(height=size, width=size),
                   A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ToTensorV2()]),
        A.Compose([A.Resize(height=size, width=size), A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ToTensorV2()]),
        A.Compose([A.RandomResizedCrop(size=(size, size), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                   A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ToTensorV2()]),
        A.Compose([A.RandomResizedCrop(size=(size, size), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                   A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ToTensorV2()]),
        A.Compose([A.Resize(height=size, width=size),
                   A.CenterCrop(height=int(size * 0.9), width=int(size * 0.9)),
                   A.Resize(height=size, width=size),
                   A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                   ToTensorV2()]),
    ]
    return tfms[:CFG.tta_steps]


class CatDogDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, is_test=False):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.is_test   = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.is_test:
            return img, row["id"]
        return img, torch.tensor(row["label"], dtype=torch.float32)


# ─────────────────────────── 模型 ────────────────────────────────────
class CatDogModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True,
                 num_classes: int = 1, drop_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # 去掉原始分类头
            drop_rate=drop_rate,
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(drop_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(1)


# ─────────────────────── MixUp / CutMix ──────────────────────────────
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y[idx], lam


def mix_criterion(pred, y_a, y_b, lam, criterion):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────── 学习率调度 ─────────────────────────────────
def get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(CFG.min_lr / CFG.lr,
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────── 训练 / 验证循环 ─────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion):
    model.train()
    losses, preds, targets = [], [], []

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs   = imgs.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)

        # MixUp / CutMix
        use_mix = random.random() < CFG.mix_prob
        if use_mix and (CFG.mixup_alpha > 0 or CFG.cutmix_alpha > 0):
            if random.random() < 0.5 and CFG.cutmix_alpha > 0:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels, CFG.cutmix_alpha)
            else:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, CFG.mixup_alpha)
            with autocast(enabled=CFG.amp):
                out  = model(imgs)
                loss = mix_criterion(out, y_a, y_b, lam, criterion)
        else:
            with autocast(enabled=CFG.amp):
                out  = model(imgs)
                loss = criterion(out, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
        preds.extend(torch.sigmoid(out.detach()).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    avg_loss = np.mean(losses)
    score    = get_score(np.array(targets) > 0.5,
                         np.clip(np.array(preds), 1e-7, 1 - 1e-7))
    return avg_loss, score


@torch.no_grad()
def valid_one_epoch(model, loader, criterion):
    model.eval()
    losses, preds, targets = [], [], []

    for imgs, labels in tqdm(loader, desc="Valid", leave=False):
        imgs   = imgs.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)
        with autocast(enabled=CFG.amp):
            out  = model(imgs)
            loss = criterion(out, labels)
        losses.append(loss.item())
        preds.extend(torch.sigmoid(out).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    avg_loss = np.mean(losses)
    score    = get_score(np.array(targets) > 0.5,
                         np.clip(np.array(preds), 1e-7, 1 - 1e-7))
    return avg_loss, score, np.array(preds)


# ──────────────────────────── 推理 (TTA) ─────────────────────────────
@torch.no_grad()
def predict_tta(model, df_test: pd.DataFrame) -> np.ndarray:
    model.eval()
    tta_tfms  = get_tta_transforms()
    all_preds = []

    for tfm in tqdm(tta_tfms, desc="TTA"):
        ds     = CatDogDataset(df_test, transform=tfm, is_test=True)
        loader = DataLoader(ds, batch_size=CFG.batch_size * 2,
                            num_workers=CFG.num_workers, pin_memory=True)
        preds  = []
        for imgs, _ in loader:
            imgs = imgs.to(CFG.device, non_blocking=True)
            with autocast(enabled=CFG.amp):
                out = torch.sigmoid(model(imgs))
            preds.extend(out.cpu().numpy())
        all_preds.append(np.array(preds))

    return np.mean(all_preds, axis=0)


# ──────────────────────── 训练曲线绘图 ───────────────────────────────
def _plot_history(history: dict):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["trn_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],  label="valid")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["trn_logloss"], label="train")
    axes[1].plot(epochs, history["val_logloss"],  label="valid")
    axes[1].set_title("LogLoss")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(CFG.output_dir / "training_curve.png", dpi=120)
    plt.close(fig)


# ──────────────────────────── 主流程 ─────────────────────────────────
def main():
    seed_everything(CFG.seed)
    print(f"Device: {CFG.device}  |  Model: {CFG.model_name}  |  Image size: {CFG.image_size}")

    # ── 构建 DataFrame，按类别分层拆分 ──
    df = build_dataframe()
    trn_df, val_df = train_test_split(
        df, test_size=CFG.val_ratio, stratify=df["label"], random_state=CFG.seed)
    trn_df = trn_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train: {len(trn_df)}  Val: {len(val_df)}  "
          f"(cat {(df.label==0).sum()}, dog {(df.label==1).sum()})")

    # ── 测试集 DataFrame ──
    test_paths = sorted(CFG.test_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    df_test = pd.DataFrame({
        "path": test_paths,
        "id":   [int(p.stem) for p in test_paths],
    })

    # ── DataLoader ──
    trn_loader = DataLoader(
        CatDogDataset(trn_df, transform=get_transforms("train")),
        batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        CatDogDataset(val_df, transform=get_transforms("valid")),
        batch_size=CFG.batch_size * 2, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True)

    # ── 模型 / 优化器 / 调度器 ──
    model     = CatDogModel(CFG.model_name, CFG.pretrained, CFG.num_classes).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    total_steps  = CFG.num_epochs * len(trn_loader)
    warmup_steps = CFG.warmup_epochs * len(trn_loader)
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler(enabled=CFG.amp)

    class SmoothBCE(nn.Module):
        def forward(self, pred, target):
            target = target * (1 - CFG.label_smoothing) + 0.5 * CFG.label_smoothing
            return F.binary_cross_entropy_with_logits(pred, target)

    criterion = SmoothBCE()

    # ── 训练循环 ──
    best_score = np.inf
    history    = {"epoch": [], "trn_loss": [], "val_loss": [],
                  "trn_logloss": [], "val_logloss": [], "lr": []}

    for epoch in range(CFG.num_epochs):
        trn_loss, trn_score = train_one_epoch(
            model, trn_loader, optimizer, scheduler, scaler, criterion)
        val_loss, val_score, _ = valid_one_epoch(model, val_loader, criterion)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:2d}/{CFG.num_epochs} | "
              f"trn_loss: {trn_loss:.4f}  trn_logloss: {trn_score:.4f} | "
              f"val_loss: {val_loss:.4f}  val_logloss: {val_score:.4f} | "
              f"lr: {cur_lr:.2e}")

        history["epoch"].append(epoch + 1)
        history["trn_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)
        history["trn_logloss"].append(trn_score)
        history["val_logloss"].append(val_score)
        history["lr"].append(cur_lr)

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), CFG.output_dir / "best_model.pth")
            print(f"  ✓ Best model saved  (val_logloss={best_score:.4f})")

        # 每个 epoch 实时更新曲线图
        _plot_history(history)
        pd.DataFrame(history).to_csv(CFG.output_dir / "history.csv", index=False)

    print(f"  ✓ Training curve saved → training_curve.png")

    # ── 用最优权重推理测试集 (TTA) ──
    print(f"\nRunning inference with TTA x{CFG.tta_steps} ...")
    model.load_state_dict(torch.load(CFG.output_dir / "best_model.pth",
                                     map_location=CFG.device))
    test_preds = predict_tta(model, df_test)

    # ── 生成提交文件 ──
    sub = pd.read_csv(CFG.data_dir / "sample_submission.csv")
    sub["label"] = test_preds.clip(1e-4, 1 - 1e-4)
    sub.to_csv(CFG.output_dir / "submission.csv", index=False)
    print(f"Submission saved → {CFG.output_dir / 'submission.csv'}")
    print(f"Best val_logloss: {best_score:.5f}")
    print(sub.head())


if __name__ == "__main__":
    main()
