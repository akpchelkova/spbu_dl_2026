"""Дообучение ConvNeXt / ViT с 5-fold разбиением и проверкой macro-F1.

Пример: python train.py --fold 0 --epochs 22
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import (PictureDataset, count_weights, freeze_except_classifier,
                    get_class_names, get_timm_model, make_transforms,
                    seed_everything, select_device, split_head_backbone_params)


def build_lr_scheduler(optimizer, epochs: int, steps_per_epoch: int):
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * 0.05))

    def multiplier(step):
        if step < warmup_steps:
            return max(0.05, (step + 1) / warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.025 + 0.975 * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler,
                    device, epoch, freeze_epochs):
    model.train()
    freeze_except_classifier(model, freeze=epoch < freeze_epochs)
    losses, seen = 0.0, 0
    correct = 0
    progress = tqdm(loader, desc=f'Train epoch {epoch + 1}', leave=False)
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        n = y.size(0)
        seen += n
        correct += (logits.argmax(1) == y).sum().item()
        losses += loss.item() * n
        progress.set_postfix(loss=f'{losses/seen:.4f}', acc=f'{correct/seen:.3f}')
    return losses / seen, correct / seen


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    preds, targets, probs = [], [], []
    for x, y in tqdm(loader, desc='Validation', leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            p = model(x).float().softmax(dim=1)
        probs.append(p.cpu().numpy())
        targets.append(y.numpy())
        preds.append(p.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    return {
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'probs': np.concatenate(probs),
        'y_true': y_true,
        'y_pred': y_pred,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--save-dir', default='checkpoints')
    p.add_argument('--model', default='convnext_tiny.fb_in22k_ft_in1k')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--full', action='store_true', help='Обучить на всех train без валидации')
    p.add_argument('--epochs', type=int, default=22)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--backbone-lr', type=float, default=6e-5)
    p.add_argument('--head-lr', type=float, default=5e-4)
    p.add_argument('--freeze-epochs', type=int, default=1)
    p.add_argument('--label-smoothing', type=float, default=0.06)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--weighting', choices=['sqrt', 'none'], default='sqrt')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed + (0 if args.full else args.fold))
    device = select_device()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    names = get_class_names(data_dir)
    n_classes = len(names)
    df = pd.read_csv(data_dir / 'train_folds.csv')
    if args.full:
        train_df = df.reset_index(drop=True)
        val_df = None
    else:
        if args.fold not in set(df.fold):
            raise ValueError(f'Недопустимый fold {args.fold}. Допустимые: {sorted(set(df.fold))}')
        train_df = df.loc[df.fold != args.fold].reset_index(drop=True)
        val_df = df.loc[df.fold == args.fold].reset_index(drop=True)
    print(f'Обучение: {len(train_df)} изображений; валидация: {0 if val_df is None else len(val_df)}')

    train_ds = PictureDataset(train_df, data_dir, make_transforms(args.img_size, training=True))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == 'cuda',
                              persistent_workers=args.workers > 0)
    if val_df is not None:
        val_ds = PictureDataset(val_df, data_dir, make_transforms(args.img_size, training=False))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=args.workers, pin_memory=device.type == 'cuda',
                                persistent_workers=args.workers > 0)

    model = get_timm_model(args.model, pretrained=True, n_classes=n_classes).to(device)
    backbone, head = split_head_backbone_params(model)
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': args.backbone_lr},
        {'params': head, 'lr': args.head_lr},
    ], weight_decay=0.04)
    scheduler = build_lr_scheduler(optimizer, args.epochs, len(train_loader))
    weights = (count_weights(train_df.label.to_numpy(), n_classes).to(device)
               if args.weighting == 'sqrt' else None)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    file_tag = re.sub('[^a-zA-Z0-9_-]+', '_', args.model)
    file_tag += '_full' if args.full else f'_fold{args.fold}'
    checkpoint = save_dir / f'{file_tag}.pt'
    metrics_path = save_dir / f'{file_tag}_history.csv'
    oof_path = save_dir / f'{file_tag}_oof.csv'
    history, best = [], -1.0
    bad_epochs = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                               scheduler, criterion, scaler, device,
                                               epoch, args.freeze_epochs)
        row = {'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc}
        if val_df is not None:
            result = evaluate(model, val_loader, device)
            row.update({k: result[k] for k in ('macro_f1', 'weighted_f1', 'accuracy')})
            print(f'Epoch {epoch+1:02d}/{args.epochs} | loss={train_loss:.4f} | '
                  f'macro-F1={result["macro_f1"]:.4f} | '
                  f'weighted-F1={result["weighted_f1"]:.4f} | '
                  f'acc={result["accuracy"]:.4f}')
            improved = result['macro_f1'] > best + 1e-5
        else:
            improved = True
            print(f'Epoch {epoch+1:02d}/{args.epochs} | loss={train_loss:.4f} | acc={train_acc:.4f}')

        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False)
        if improved:
            if val_df is not None:
                best = result['macro_f1']
            bad_epochs = 0
            # State dict — стандартный переносимый PyTorch checkpoint.
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'model_name': args.model, 'img_size': args.img_size,
                        'class_names': names, 'fold': -1 if args.full else args.fold,
                        'epoch': epoch + 1, 'best_macro_f1': best}, checkpoint)
            if val_df is not None:
                out = val_df[['id', 'label']].copy()
                for j in range(n_classes):
                    out[f'p{j}'] = result['probs'][:, j]
                out.to_csv(oof_path, index=False)
                print('  New best! Модель сохранена:', checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f'Ранняя остановка: {args.patience} эпох без улучшений')
                break

    print('Готово. Лучшая модель:', checkpoint.resolve())
    if val_df is not None:
        print('Лучшая macro-F1:', round(best, 5))
        oof = pd.read_csv(oof_path)
        pred = oof[[f'p{i}' for i in range(n_classes)]].to_numpy().argmax(axis=1)
        print(classification_report(oof.label.to_numpy(), pred, labels=list(range(n_classes)),
                                    target_names=names, zero_division=0))


if __name__ == '__main__':
    main()
