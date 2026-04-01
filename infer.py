import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CFG:
    data_dir   = Path("/data/smx/catordog/dataset")
    test_dir   = data_dir / "test"
    output_dir = Path("/data/smx/catordog/output")

    model_name  = "convnext_large_in22ft1k"
    image_size  = 384
    batch_size  = 64
    num_workers = 12
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    amp         = True


class CatDogDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, row["id"]


class CatDogModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = False,
                 num_classes: int = 1, drop_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
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


def get_safe_tta_transforms():
    size = CFG.image_size
    return [
        A.Compose([
            A.Resize(height=size, width=size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=size, width=size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=size, width=size),
            A.CenterCrop(height=int(size * 0.9), width=int(size * 0.9)),
            A.Resize(height=size, width=size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]


@torch.no_grad()
def predict_tta(model, df_test):
    model.eval()
    all_preds = []

    for tfm in get_safe_tta_transforms():
        ds = CatDogDataset(df_test, transform=tfm)
        loader = DataLoader(
            ds,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True
        )

        preds = []
        for imgs, _ in loader:
            imgs = imgs.to(CFG.device, non_blocking=True)
            with autocast(enabled=CFG.amp):
                out = torch.sigmoid(model(imgs))
            preds.extend(out.cpu().numpy())
        all_preds.append(np.array(preds))

    return np.mean(all_preds, axis=0)


def main():
    test_paths = sorted(CFG.test_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    df_test = pd.DataFrame({
        "path": test_paths,
        "id": [int(p.stem) for p in test_paths],
    })

    model = CatDogModel(CFG.model_name, pretrained=False).to(CFG.device)
    ckpt_path = CFG.output_dir / "best_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=CFG.device))

    preds = predict_tta(model, df_test)

    sub = pd.read_csv(CFG.data_dir / "sample_submission.csv")
    sub["label"] = preds.clip(1e-4, 1 - 1e-4)
    save_path = CFG.output_dir / "submission_safe_tta.csv"
    sub.to_csv(save_path, index=False)

    print(f"Saved to: {save_path}")
    print(sub.head())


if __name__ == "__main__":
    main()