"""Общие функции для One Piece Kaggle competition."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('Device: Apple MPS')
        return torch.device('mps')
    print('ВНИМАНИЕ: CUDA недоступна. Используется CPU; обучение будет медленным.')
    return torch.device('cpu')


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as orig:
        img = ImageOps.exif_transpose(orig)
        if img.mode in ('RGBA', 'LA') or ('transparency' in img.info):
            img = img.convert('RGBA')
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            background.alpha_composite(img)
            return background.convert('RGB')
        return img.convert('RGB')


class SquarePad:
    """Сохраняем пропорции. Особенно важно для аниме-кадров 16:9."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = max(w, h)
        left = (side - w) // 2
        top = (side - h) // 2
        return ImageOps.expand(img, (left, top, side - w - left, side - h - top), fill=(128, 128, 128))


def make_transforms(size: int, training: bool, augmentation: bool = True):
    if training and augmentation:
        return T.Compose([
            SquarePad(),
            T.RandomResizedCrop(size, scale=(0.80, 1.0), ratio=(0.94, 1.06),
                                interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.16, contrast=0.16,
                                         saturation=0.12, hue=0.02)], p=0.7),
            T.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.96, 1.04),
                           fill=(128, 128, 128), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.10, scale=(0.015, 0.07), ratio=(0.5, 2.0)),
        ])
    return T.Compose([
        SquarePad(),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


class PictureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: Path, transform, labeled: bool = True):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.transform(open_rgb(self.data_dir / row['relpath']))
        if self.labeled:
            return img, int(row['label'])
        return img


def count_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes)
    # Мягкая компенсация дисбаланса, чтобы не переобучать редкие классы.
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def get_class_names(data_dir: Path) -> list[str]:
    import json
    class_map = json.loads((Path(data_dir) / 'labels.json').read_text(encoding='utf-8'))
    return [class_map[str(i)] for i in range(len(class_map))]


def freeze_except_classifier(model: torch.nn.Module, freeze: bool) -> None:
    if not freeze:
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def split_head_backbone_params(model):
    head_params = list(model.get_classifier().parameters())
    head_ids = set(map(id, head_params))
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    assert head_params and backbone_params, 'Не удалось определить слои классификации'
    return backbone_params, head_params


def get_timm_model(name: str, pretrained: bool, n_classes: int):
    try:
        import timm
    except ImportError as e:
        raise SystemExit('Установите зависимости: python -m pip install -r requirements.txt') from e
    return timm.create_model(name, pretrained=pretrained, num_classes=n_classes)
