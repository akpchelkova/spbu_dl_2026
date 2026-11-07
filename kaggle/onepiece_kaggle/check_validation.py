"""Проверка macro-F1 одного фолда и ансамбля ОДНОГО фолда (без leakage).

Пример: python check_validation.py --oof checkpoints/model_a_fold0_oof.csv checkpoints/model_b_fold0_oof.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from common import get_class_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--oof', nargs='+', required=True)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output', default='outputs/validation_errors.csv')
    args = p.parse_args()
    names = get_class_names(Path(args.data_dir))
    columns = [f'p{i}' for i in range(len(names))]
    dfs = [pd.read_csv(path) for path in args.oof]
    # Сравниваем строго одни и те же ID, никогда не пересекаем разные CV-фолды.
    first = dfs[0].set_index('id').sort_index()
    for df in dfs[1:]:
        second = df.set_index('id').sort_index()
        if not first.index.equals(second.index) or not first.label.equals(second.label):
            raise ValueError('OOF файлы должны иметь те же ID и true label: объедините модели одного fold!')
    avg = sum(df.set_index('id').sort_index()[columns].to_numpy() for df in dfs) / len(dfs)
    preds = avg.argmax(axis=1)
    labels = first.label.to_numpy()
    print('macro-F1', round(f1_score(labels, preds, average='macro', zero_division=0), 5))
    print('weighted-F1', round(f1_score(labels, preds, average='weighted', zero_division=0), 5))
    print('accuracy', round(accuracy_score(labels, preds), 5))
    print(classification_report(labels, preds, labels=list(range(len(names))),
                                target_names=names, zero_division=0))
    out = pd.DataFrame({'id': first.index, 'true_label': labels, 'prediction': preds,
                        'confidence': avg.max(axis=1)})
    out['true_name'] = [names[i] for i in labels]
    out['predicted_name'] = [names[i] for i in preds]
    errors = out[out.true_label != out.prediction].sort_values('confidence', ascending=False)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(dest, index=False)
    print('Ошибки валидации:', len(errors), 'файл:', dest)


if __name__ == '__main__':
    main()
