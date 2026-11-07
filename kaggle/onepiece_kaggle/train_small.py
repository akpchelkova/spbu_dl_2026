"""Обучение своей компактной CNN с нуля; дистилляция опциональна.

Без учителя: python train_small.py --fold 0 --epochs 90
С учителем: python train_small.py --fold 0 --teacher-checkpoint checkpoints/...
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import PictureDataset, count_weights, get_class_names, make_transforms, seed_everything, select_device
from small_model import SmallAnimeCNN
from train import build_lr_scheduler, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--save-dir', default='checkpoints')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--epochs', type=int, default=90)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--img-size', type=int, default=160)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=22)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--teacher-checkpoint', default=None,
                   help='Необязательно: готовая сильная модель для knowledge distillation')
    p.add_argument('--distill-alpha', type=float, default=0.35)
    p.add_argument('--distill-temperature', type=float, default=2.5)
    args = p.parse_args()
    seed_everything(args.seed + args.fold)
    device = select_device()
    root = Path(args.data_dir)
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / 'train_folds.csv')
    names = get_class_names(root)
    tr = df[df.fold != args.fold].reset_index(drop=True)
    va = df[df.fold == args.fold].reset_index(drop=True)
    train_loader = DataLoader(PictureDataset(tr, root, make_transforms(args.img_size, True)),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              persistent_workers=args.workers > 0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(PictureDataset(va, root, make_transforms(args.img_size, False)),
                            batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers,
                            persistent_workers=args.workers > 0, pin_memory=device.type == 'cuda')

    model = SmallAnimeCNN(len(names)).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f'Наша модель: {parameter_count:,} параметров (~ {parameter_count * 4 / 1024**2:.2f} MiB FP32)')
    teacher = None
    teacher_size = None
    if args.teacher_checkpoint:
        from predict import load_checkpoint
        teacher, metadata = load_checkpoint(args.teacher_checkpoint, device)
        teacher_size = int(metadata['img_size'])
        if metadata['class_names'] != names:
            raise ValueError('Несовместимы номера классов teacher/student')
        if metadata['fold'] != args.fold:
            raise ValueError('Для честной валидации учитель и ученик должны обучаться на одном fold')
        teacher.requires_grad_(False)
        print('Дистилляция от:', metadata['model_name'], 'fold=', metadata['fold'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.025)
    scheduler = build_lr_scheduler(optimizer, args.epochs, len(train_loader))
    criterion = nn.CrossEntropyLoss(weight=count_weights(tr.label.to_numpy(), len(names)).to(device),
                                    label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    ckpt_path = save / f'custom_tiny_cnn_fold{args.fold}.pt'
    history_path = save / f'custom_tiny_cnn_fold{args.fold}_history.csv'
    oof_path = save / f'custom_tiny_cnn_fold{args.fold}_oof.csv'
    best, bad_epochs, history = -1., 0, []

    for epoch in range(args.epochs):
        model.train()
        seen, total_loss = 0, 0.
        for x, y in tqdm(train_loader, desc=f'Small CNN epoch {epoch+1}', leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                logits = model(x)
                ce = criterion(logits, y)
                loss = ce
                if teacher is not None:
                    with torch.no_grad():
                        x_teacher = F.interpolate(x, size=(teacher_size, teacher_size),
                                                  mode='bilinear', align_corners=False)
                        teacher_logits = teacher(x_teacher)
                    temperature = args.distill_temperature
                    kd = F.kl_div(
                        F.log_softmax(logits.float() / temperature, dim=1),
                        F.softmax(teacher_logits.float() / temperature, dim=1),
                        reduction='batchmean') * (temperature ** 2)
                    loss = (1 - args.distill_alpha) * ce + args.distill_alpha * kd
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            seen += len(x)
            total_loss += float(loss.item()) * len(x)

        res = evaluate(model, val_loader, device)
        epoch_info = {'epoch': epoch + 1, 'loss': total_loss / seen,
                      'macro_f1': res['macro_f1'], 'weighted_f1': res['weighted_f1'],
                      'accuracy': res['accuracy']}
        history.append(epoch_info)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f'Epoch {epoch+1}/{args.epochs}: loss={epoch_info["loss"]:.4f}, '
              f'macro-F1={res["macro_f1"]:.4f}, acc={res["accuracy"]:.4f}')
        if res['macro_f1'] > best + 1e-5:
            best, bad_epochs = res['macro_f1'], 0
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'model_name': 'custom_tiny_cnn', 'class_names': names,
                        'img_size': args.img_size, 'fold': args.fold,
                        'epoch': epoch + 1, 'best_macro_f1': best}, ckpt_path)
            oof = va[['id', 'label']].copy()
            for j in range(len(names)):
                oof[f'p{j}'] = res['probs'][:, j]
            oof.to_csv(oof_path, index=False)
            size_mb = ckpt_path.stat().st_size / 1e6
            print(f'  Лучшая модель сохранена. Размер файла: {size_mb:.2f} MB')
            if size_mb >= 5:
                raise AssertionError('Размер модели превысил 5 MB, надо уменьшить архитектуру')
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print('Ранняя остановка')
            break
    print('Лучший F1:', best, 'файл:', ckpt_path)


if __name__ == '__main__':
    main()
