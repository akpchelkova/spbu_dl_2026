"""HTML для ручной проверки 62 близких совпадений train/test.

Запуск: python review_matches.py
Откройте outputs/review_duplicates.html в браузере, отметьте подтверждённые пары,
нажмите «Скачать подтверждённые» и сохраните как outputs/manual_overrides.csv.
"""
from __future__ import annotations

import argparse
import base64
import html
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from common import open_rgb


def thumbnail_datauri(path: Path) -> str:
    im = open_rgb(path)
    im.thumbnail((300, 220), Image.Resampling.LANCZOS)
    buf = BytesIO()
    im.save(buf, 'JPEG', quality=78)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output', default='outputs/review_duplicates.html')
    args = p.parse_args()
    data = Path(args.data_dir)
    df = pd.read_csv(data / 'near_duplicate_candidates.csv')
    sections = []
    for item in df.sort_values(['distance', 'candidate_name']).itertuples():
        test_img = thumbnail_datauri(data / item.test_image)
        train_img = thumbnail_datauri(data / item.train_image)
        sections.append(f'''
        <article class="entry">
          <div class="info"><label><input type="checkbox" class="confirm" data-id="{html.escape(str(item.id))}"
                   data-label="{int(item.candidate_label)}"> Подтвердить совпадение</label>
              <span>pHash расстояние: {int(item.distance)}</span>
              <strong>Предлагаемый класс: {html.escape(str(item.candidate_name))} ({int(item.candidate_label)})</strong>
              <small>Следующий кандидат: расстояние {int(item.second_distance)}, класс {int(item.second_label)}</small>
          </div>
          <div class="pics"><figure><img src="{test_img}"><figcaption>TEST: {html.escape(str(item.id))}</figcaption></figure>
          <figure><img src="{train_img}"><figcaption>TRAIN: {html.escape(str(item.train_image))}</figcaption></figure></div>
        </article>''')
    page = '''<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
    <title>Проверка близких картинок One Piece</title><style>
    body{font-family:system-ui,Arial,sans-serif;background:#f3f6fa;color:#172432;max-width:1100px;margin:24px auto;padding:0 18px}
    h1{font-size:24px}.hint{line-height:1.5}.entry{background:white;border:1px solid #dde4ee;border-radius:10px;padding:14px;margin:12px 0}
    .info{display:flex;gap:16px;align-items:center;flex-wrap:wrap}.info small{color:#5a6877}
    .pics{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
    figure{margin:0}img{max-width:100%;height:220px;object-fit:contain;background:#eee}figcaption{word-break:break-all;font-size:12px;color:#566;padding-top:4px}
    #export{cursor:pointer;background:#174ab5;border:0;border-radius:8px;padding:12px 18px;color:white;font-weight:bold}
    .sticky{position:sticky;top:0;background:#f3f6fa;padding:12px 0;z-index:1}
    </style></head><body><h1>Проверка похожих изображений</h1>
    <p class="hint">Отмечайте только пары, где на TEST и TRAIN изображён один и тот же персонаж.
    Низкое расстояние pHash ещё НЕ доказывает совпадение класса. Точные совпадения по SHA256 код уже учитывает автоматически.</p>
    <div class="sticky"><button id="export">Скачать подтверждённые совпадения CSV</button> <span id="count">Выбрано: 0</span></div>
    ''' + '\n'.join(sections) + '''
    <script>
    const count = document.querySelector('#count');
    document.addEventListener('change', () => {
      count.textContent = 'Выбрано: ' + document.querySelectorAll('.confirm:checked').length;
    });
    document.querySelector('#export').addEventListener('click', () => {
      const selected = [...document.querySelectorAll('.confirm:checked')];
      const csv = 'id,label\\n' + selected.map(x => x.dataset.id + ',' + x.dataset.label).join('\\n') + '\\n';
      const link = document.createElement('a');
      link.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
      link.download = 'manual_overrides.csv'; link.click();
      URL.revokeObjectURL(link.href);
    });
    </script></body></html>'''
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding='utf-8')
    print('Создан HTML:', out.resolve())
    print('Кандидатов для проверки:', len(df))


if __name__ == '__main__':
    main()
