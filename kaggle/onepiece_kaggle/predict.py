"""Ансамбль обученных моделей, TTA и Kaggle submission.

Пример: python predict.py --checkpoints checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import PictureDataset, get_class_names, get_timm_model, make_transforms, select_device


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt['model_name'] == 'custom_tiny_cnn':
        from small_model import SmallAnimeCNN
        model = SmallAnimeCNN(num_classes=len(ckpt['class_names']))
    else:
        model = get_timm_model(ckpt['model_name'], pretrained=False, n_classes=len(ckpt['class_names']))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model.to(device).eval(), ckpt


@torch.inference_mode()
def predict_single(model, loader, device, tta: int):
    all_probabilities = []
    for x in tqdm(loader, desc='Prediction', leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            p = model(x).float().softmax(dim=1)
            if tta == 2:
                flipped = model(x.flip(dims=[3])).float().softmax(dim=1)
                p = (p + flipped) / 2
        all_probabilities.append(p.cpu().numpy())
    return np.concatenate(all_probabilities)


def exact_match_predictions(train_df, test_df, n_classes):
    same_sha = train_df.groupby('sha256')['label'].agg(lambda x: list(set(x)))
    known = {sha: labels[0] for sha, labels in same_sha.items() if len(labels) == 1}
    matches = {i: known[sha] for i, sha in enumerate(test_df.sha256) if sha in known}
    print(f'Точных совпадений test/train с однозначной меткой: {len(matches)}')
    return matches


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=str(Path(__file__).resolve().parent / 'data'))
    p.add_argument('--checkpoints', required=True, nargs='+')
    p.add_argument('--output', default=str(Path(__file__).resolve().parent / 'outputs' / 'submission.csv'))
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--tta', type=int, choices=(1, 2), default=2)
    p.add_argument('--no-exact-match', action='store_true')
    p.add_argument('--overrides', default=None, help='Подтверждённые вручную пары: CSV id,label')
    args = p.parse_args()
    device = select_device()
    data_dir = Path(args.data_dir)
    test_df = pd.read_csv(data_dir / 'test_manifest.csv', dtype={'id': 'string'})
    names = get_class_names(data_dir)
    sum_probs = None
    sum_weight = 0.0
    print(f'Тестовых файлов: {len(test_df)}, чекпоинтов: {len(args.checkpoints)}')
    for idx, checkpoint in enumerate(args.checkpoints, start=1):
        model, meta = load_checkpoint(checkpoint, device)
        if meta['class_names'] != names:
            raise ValueError(f'Неверное соответствие меток в {checkpoint}')
        ds = PictureDataset(test_df, data_dir, make_transforms(meta['img_size'], training=False),
                            labeled=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == 'cuda',
                            persistent_workers=args.workers > 0)
        print(f'[{idx}] {meta["model_name"]} / fold {meta["fold"]} / img={meta["img_size"]}')
        probabilities = predict_single(model, loader, device, args.tta)
        if sum_probs is None:
            sum_probs = np.zeros_like(probabilities, dtype=np.float64)
        sum_probs += probabilities
        sum_weight += 1.0
        del model, loader, ds
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    probs = sum_probs / sum_weight
    prediction = probs.argmax(axis=1).astype(int)
    if not args.no_exact_match:
        train_df = pd.read_csv(data_dir / 'train_folds.csv')
        matches = exact_match_predictions(train_df, test_df, n_classes=len(names))
        for index, label in matches.items():
            prediction[index] = label
            probs[index] = 0
            probs[index, label] = 1.0

    if args.overrides is not None:
        manual = pd.read_csv(args.overrides, dtype={'id': 'string'})
        if not {'id', 'label'}.issubset(manual.columns):
            raise ValueError('В ручном CSV нужны столбцы id,label')
        if manual.id.duplicated().any() or not set(manual.id).issubset(set(test_df.id)):
            raise ValueError('Есть несуществующие или повторяющиеся test ID в overrides')
        if manual.label.isna().any() or not manual.label.between(0, len(names) - 1).all():
            raise ValueError('В overrides есть неверные номера классов')
        idx_by_id = {sample_id: i for i, sample_id in enumerate(test_df.id)}
        for r in manual.itertuples():
            index, label = idx_by_id[r.id], int(r.label)
            prediction[index] = label
            probs[index] = 0
            probs[index, label] = 1.0
        print(f'Применены подтверждённые вручную совпадения: {len(manual)}')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({'id': test_df.id, 'label': prediction})
    result.to_csv(output_path, index=False)
    probability_df = pd.DataFrame(probs, columns=[f'p{i}' for i in range(len(names))])
    probability_df.insert(0, 'id', test_df.id)
    probability_df['confidence'] = probs.max(axis=1)
    probability_df['predicted_name'] = [names[i] for i in prediction]
    probability_df.sort_values('confidence').to_csv(
        output_path.with_name(output_path.stem + '_probabilities.csv'), index=False)
    print('Submission:', output_path.resolve())
    print('Таблица уверенности:', output_path.with_name(output_path.stem + '_probabilities.csv'))
    print('Распределение классов:')
    print(result.label.map(dict(enumerate(names))).value_counts().to_string())
    print('Строк:', len(result), 'уникальных ID:', result.id.nunique())
    assert len(result) == 849 and result.id.is_unique
    assert result.label.between(0, len(names) - 1).all()


if __name__ == '__main__':
    main()
