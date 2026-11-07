"""Распаковка, проверка разметки, защита от дублей, групповой CV-split.

Запуск: python prepare_data.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

from common import open_rgb

IMAGES = {'.jpg', '.jpeg', '.png', '.webp'}


def perceptual_hash(path: Path) -> int:
    image = ImageOps.grayscale(open_rgb(path)).resize((32, 32), Image.Resampling.LANCZOS)
    data = np.asarray(image, dtype=np.float32)
    spectrum = dctn(data, norm='ortho')[:8, :8].reshape(-1)
    median = np.median(spectrum[1:])
    return sum(int(value > median) << i for i, value in enumerate(spectrum))


def extract_images(archive: Path, destination: Path, is_train: bool) -> dict[str, str]:
    result = {}
    with zipfile.ZipFile(archive) as z:
        bad_file = z.testzip()
        if bad_file is not None:
            raise RuntimeError(f'Повреждён архив {archive}: {bad_file}')
        for member in z.infolist():
            parts = Path(member.filename.replace('\\', '/')).parts
            if member.is_dir() or Path(member.filename).suffix.lower() not in IMAGES:
                continue
            if is_train:
                if len(parts) != 3:
                    raise ValueError(f'Неожиданный путь в train-архиве: {member.filename}')
                cls, filename = parts[-2:]
                target = destination / 'train' / cls / filename
                key = f'train/{cls}/{filename}'
            else:
                if len(parts) != 2:
                    raise ValueError(f'Неожиданный путь в test-архиве: {member.filename}')
                filename = parts[-1]
                target = destination / 'test' / filename
                key = Path(filename).stem
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size != member.file_size:
                with z.open(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
            if key in result:
                raise RuntimeError(f'Дублирующийся путь/ID в архивах: {key}')
            result[key] = str(target.relative_to(destination)).replace('\\', '/')
    return result


def union_find_groups(df: pd.DataFrame, threshold: int = 4) -> list[str]:
    """Перцептуально близкие изображения одного класса держим в одном фолде."""
    result = [''] * len(df)
    for cls, group in df.groupby('class', sort=True):
        indices = list(group.index)
        hashes = [int(df.at[i, 'phash'], 16) for i in indices]
        parent = list(range(len(indices)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if (hashes[i] ^ hashes[j]).bit_count() <= threshold:
                    a, b = find(i), find(j)
                    parent[a] = b
        for i, global_idx in enumerate(indices):
            result[global_idx] = f'{cls}_{find(i)}'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='raw')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    raw, data = Path(args.raw_dir), Path(args.data_dir)
    required = ['train_first_part.zip', 'train_second_part.zip', 'test.zip',
                'train_annotations.csv', 'submission.csv', 'labels.json']
    missing = [s for s in required if not (raw / s).is_file()]
    if missing:
        raise FileNotFoundError(f'Скопируйте в {raw.resolve()} файлы: {missing}')
    data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw / 'labels.json', data / 'labels.json')
    labels = json.loads((raw / 'labels.json').read_text(encoding='utf-8'))

    extracted = {}
    for name in ('train_first_part.zip', 'train_second_part.zip'):
        current = extract_images(raw / name, data, is_train=True)
        intersection = set(extracted) & set(current)
        if intersection:
            raise ValueError(f'Совпадающие имена изображений в частях архива: {intersection}')
        extracted.update(current)
    tests = extract_images(raw / 'test.zip', data, is_train=False)

    df = pd.read_csv(raw / 'train_annotations.csv', dtype={'id': 'string'})
    if df['image_path'].duplicated().any():
        duplicates = df[df['image_path'].duplicated(keep=False)]
        if duplicates.groupby('image_path')['label'].nunique().max() > 1:
            raise ValueError('Есть противоречащие метки у одного изображения!')
        print(f'Удалены повторные записи CSV: {df.image_path.duplicated().sum()}')
    df = df.drop_duplicates('image_path').reset_index(drop=True)
    df['relpath'] = df['image_path'].str.replace('\\', '/', regex=False)
    if set(df.relpath) != set(extracted):
        print('Нет файлов:', sorted(set(df.relpath)-set(extracted))[:5])
        print('Нет разметки:', sorted(set(extracted)-set(df.relpath))[:5])
        raise ValueError('Разметка и фактические изображения не совпадают')
    for i, r in df.iterrows():
        if labels[str(r['label'])] != r['class']:
            raise ValueError(f'Несоответствие имени класса и метки в строке {i}')

    sub = pd.read_csv(raw / 'submission.csv', dtype={'id': 'string'})
    if len(sub) != len(set(sub.id)) or set(sub.id) != set(tests):
        raise ValueError('ID файла submission.csv и test.zip различаются')
    tdf = pd.DataFrame({'id': sub.id.to_list(), 'relpath': [tests[s] for s in sub.id]})

    for table, name in [(df, 'train'), (tdf, 'test')]:
        print(f'Считаю хеши {name}: {len(table)} изображений')
        shas, phashes = [], []
        for relpath in table.relpath:
            path = data / relpath
            shas.append(sha256(path.read_bytes()).hexdigest())
            phashes.append(format(perceptual_hash(path), '016x'))
        table['sha256'] = shas
        table['phash'] = phashes

    # Фиксируем «похожие» группы только в пределах класса, чтобы не переоценить CV.
    df['group'] = union_find_groups(df)
    folds = np.full(len(df), -1, dtype=np.int32)
    splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True,
                                   random_state=args.seed)
    for fold, (_, val_idx) in enumerate(splitter.split(df, y=df['label'], groups=df['group'])):
        folds[val_idx] = fold
    assert (folds >= 0).all()
    df['fold'] = folds
    df.to_csv(data / 'train_folds.csv', index=False)
    tdf.to_csv(data / 'test_manifest.csv', index=False)

    # Отдельно показываем кандидатов в дубли теста (НЕ назначаем метки по близкому хешу).
    train_hashes = [(int(h, 16), int(label), relpath)
                    for h, label, relpath in zip(df.phash, df.label, df.relpath)]
    near_rows = []
    for row in tdf.itertuples():
        h = int(row.phash, 16)
        distances = sorted(((h ^ ph).bit_count(), label, path)
                           for ph, label, path in train_hashes)[:3]
        if distances[0][0] <= 6:
            near_rows.append({'id': row.id, 'distance': distances[0][0],
                              'candidate_label': distances[0][1],
                              'candidate_name': labels[str(distances[0][1])],
                              'train_image': distances[0][2],
                              'test_image': row.relpath,
                              'second_distance': distances[1][0],
                              'second_label': distances[1][1]})
    pd.DataFrame(near_rows, columns=['id', 'distance', 'candidate_label',
                                     'candidate_name', 'train_image', 'test_image',
                                     'second_distance', 'second_label']).to_csv(
                                         data / 'near_duplicate_candidates.csv', index=False)

    print(f'OK! train={len(df)}, test={len(tdf)}, классов={len(labels)}')
    print('Количество примеров по классам:')
    print(df['class'].value_counts().sort_index().to_string())
    print('Размеры валидационных фолдов:', df.fold.value_counts().sort_index().to_dict())
    print('Групп близких изображений:', df.group.nunique())
    print('Сильные совпадения для ручной проверки:', len(near_rows))
    print('Созданы: data/train_folds.csv, data/test_manifest.csv,')
    print('         data/near_duplicate_candidates.csv')


if __name__ == '__main__':
    main()
