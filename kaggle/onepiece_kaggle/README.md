# One Piece — классификация персонажей (SPbU DL / Kaggle)

Готовый проект для VS Code / Python. В нём два независимых пути:

1. **Максимально высокий Kaggle F1:** transfer learning из `timm` (ConvNeXt Tiny, при необходимости ViT Base), 5-fold валидация, аугментации, TTA, ансамбль, поиск дубликатов train/test.
2. **Дополнительные баллы:** собственная `SmallAnimeCNN` с нуля, `469 050` параметров, checkpoint в тесте занимал около **1,97 МБ** (меньше 5 МБ). Опционально её можно улучшать дистилляцией от сильной модели.

## 0. Что проверено на исходном датасете

- 18 классов: `Ace`, `Akainu`, `Brook`, `Chopper`, `Crocodile`, `Franky`, `Jinbei`, `Kurohige`, `Law`, `Luffy`, `Mihawk`, `Nami`, `Rayleigh`, `Robin`, `Sanji`, `Shanks`, `Usopp`, `Zoro`.
- В `train_annotations.csv` — 2916 строк, 2915 уникальных изображений. Запись `Jinbei/189.png` встречалась дважды; обработка убирает повтор.
- В `test.zip` — 849 изображений, и все 849 ID присутствуют в `submission.csv`.
- Разметка train и имена файлов согласованы. Все изображения успешно читаются.
- Есть 10 **полностью идентичных** изображений в train и test (по SHA256); для них прогноз можно надёжно взять из train.
- Есть 62 кандидата с близкими перцептуальными хешами (включая точные совпадения). Остальные кандидаты требуют ручной проверки.
- В исходных архивах встречаются RGB, RGBA, полутоновые и палитровые изображения; код приводит их к RGB, прозрачность заменяется на белый фон.
- `train` + `validation` делятся на пять фолдов с учётом групп похожих изображений, чтобы предотвратить утечку между подвыборками.
- **Никакой реальный F1 и Kaggle leaderboard не заявлены:** большие модели здесь не обучались, т.к. среда без видеокарты. Пайплайн был проверен на реальных файлах; маленькая сеть прошла тестовый шаг обучения и предсказания.

## 1. Содержимое проекта

```text
onepiece_kaggle/
│
├── raw/                           ← СЮДА вручную положить шесть исходных файлов
│   ├── train_first_part.zip
│   ├── train_second_part.zip
│   ├── test.zip
│   ├── train_annotations.csv
│   ├── submission.csv
│   └── labels.json
│
├── data/                          ← создаётся prepare_data.py
│   ├── train/                      ← 18 папок персонажей
│   ├── test/                       ← 849 картинок
│   ├── train_folds.csv             ← пути, метки, phash, sha256, номер fold
│   ├── test_manifest.csv
│   ├── near_duplicate_candidates.csv
│   └── labels.json
│
├── checkpoints/                   ← здесь сохраняются веса моделей и метрики
├── outputs/                       ← Kaggle CSV, таблица вероятностей, HTML-проверка
│
├── common.py                      ← преобразования, загрузка картинок, общий код
├── prepare_data.py                ← объединяет два ZIP с train, проверяет CSV, делает CV
├── train.py                       ← ConvNeXt / ViT transfer learning
├── predict.py                     ← ансамбль + TTA + Kaggle submission
├── small_model.py                 ← собственная компактная CNN
├── train_small.py                 ← обучение собственной CNN
├── check_validation.py           ← метрики ансамбля на одном validation fold
├── review_matches.py             ← HTML-каталог визуальных дублей
├── requirements.txt
└── .vscode/launch.json            ← запуск из VS Code через Run and Debug
```

## 2. Python и зависимости в VS Code (Windows, PowerShell)

**Рекомендуемый Python — 3.11.** Откройте папку `onepiece_kaggle` через VS Code (`File → Open Folder`), затем `Terminal → New Terminal`. Все дальнейшие команды запускайте **из корня** этой папки. Замените `py -3.11` на свой интерпретатор, если Python 3.11 уже выбран.

Создать виртуальное окружение (один раз):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**PyTorch с CUDA:** если есть NVIDIA GPU, откройте https://pytorch.org/get-started/locally/ и скопируйте **актуальную** команду установки PyTorch для *Windows + Pip + CUDA*. Выполните её в виртуальном окружении, заменив `pip` на ` .\.venv\Scripts\python.exe -m pip `. Нельзя гарантировать, что произвольная версия `torch` из обычного PyPI окажется CUDA-сборкой.

После PyTorch установите остальное:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Если видеокарты NVIDIA нет, допускается CPU-установка `torch` и `torchvision`, но обучение будет медленным. Проверить установку:

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision, timm; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'нет')"
```

Если `CUDA: False` при наличии NVIDIA, проверьте драйвер `nvidia-smi`, подходящую CUDA-сборку PyTorch и выбранный Python-интерпретатор. В VS Code выберите `.venv` через **Ctrl+Shift+P → Python: Select Interpreter**.

## 3. Куда положить исходные файлы

Скопируйте **ровно 6 вложений** (`train_first_part.zip`, `train_second_part.zip`, `test.zip`, `train_annotations.csv`, `submission.csv`, `labels.json`) **без распаковки** в папку **`raw`**, как показано в структуре выше. `submission.csv` из исходников является *шаблоном с нулевыми предсказаниями*, а не финальным результатом.

## 4. Объединяем ZIP, проверяем метки и создаём фолды

```powershell
.\.venv\Scripts\python.exe prepare_data.py
```

Скрипт извлечёт `train_first_part.zip` и `train_second_part.zip` в общую `data/train/<персонаж>/`, а `test.zip` — в `data/test/`. Он сверит все имена изображений с `train_annotations.csv`, удалит одну повторную строку, посчитает SHA256 и pHash и создаст **пять validation-фолдов**.

Должно появиться примерно следующее:

```text
Удалены повторные записи CSV: 1
OK! train=2915, test=849, классов=18
Размеры валидационных фолдов: {0: 583, 1: 583, 2: 583, 3: 582, 4: 584}
```

Не меняйте разбиение после подбора моделей: тогда результаты F1 останутся сопоставимыми.

## 5. Самый первый эксперимент: ConvNeXt Tiny

Сначала обучите только **fold 0** — так будет первая честная оценка результата:

```powershell
.\.venv\Scripts\python.exe train.py --model convnext_tiny.fb_in22k_ft_in1k --fold 0 --epochs 22 --batch-size 16 --workers 2
```

- `--fold 0`: около 80% датасета участвует в обучении, 20% не используется для обновления весов и оценивает F1.
- `--epochs 22`: максимальное число проходов по данным; работает ранняя остановка после 7 эпох без улучшения.
- `--batch-size 16`: размер мини-батча. При **CUDA out of memory** уменьшите до `8`, а затем до `4`.
- `--workers 2`: загрузчики изображений; на Windows при проблемах попробуйте `--workers 0`.
- `--img-size 224`: размер входа. Для ConvNeXt можно попробовать `288` и уменьшить batch-size.
- `--backbone-lr 6e-5`, `--head-lr 5e-4`: основная сеть учится осторожно, новый классификатор — быстрее.
- `--freeze-epochs 1`: в первую эпоху обучаем только классификатор, затем размораживаем всю сеть.

Веса `ConvNeXt Tiny` с ImageNet-22k/ImageNet-1k загружаются **автоматически по интернету** при первом запуске. Они не входят в проект. Их исходное описание: https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k

После обучения создаются:

```text
checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0.pt
checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0_history.csv
checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0_oof.csv
```

Смотрите на **`macro-F1` в консоли**, а также `weighted-F1` и `accuracy`. Точная разновидность усреднения F1 в тексте соревнования не указана, поэтому отображаются оба варианта. Если train loss улучшается, а validation F1 падает — это признак переобучения, а не роста качества.

## 6. Улучшаем результат: больше фолдов ConvNeXt

Когда убедились, что fold 0 успешно обучился, выполните следующие команды **по очереди**:

```powershell
.\.venv\Scripts\python.exe train.py --model convnext_tiny.fb_in22k_ft_in1k --fold 1 --epochs 22 --batch-size 16 --workers 2
.\.venv\Scripts\python.exe train.py --model convnext_tiny.fb_in22k_ft_in1k --fold 2 --epochs 22 --batch-size 16 --workers 2
```

Для ещё более устойчивого ансамбля можно обучить `--fold 3` и `--fold 4`. У каждого фолда другой набор validation-картинок, а модель предсказывает test полностью. Снижение разброса ансамбля часто полезно, но **улучшение Kaggle F1 не гарантировано**.

При сравнении разных алгоритмов сравнивайте **один и тот же fold**. Нельзя напрямую считать F1, усреднив строки разных validation-фолдов по несовпадающим ID.

## 7. Дополнительная сильная архитектура: ViT Base

Чтобы добавить разнообразие ансамблю, можно обучить Transformer, а не ещё одну CNN:

```powershell
.\.venv\Scripts\python.exe train.py --model vit_base_patch16_224.augreg_in21k_ft_in1k --fold 0 --epochs 18 --batch-size 8 --workers 2 --backbone-lr 2e-5 --head-lr 3e-4
```

`ViT Base` значительно тяжелее ConvNeXt Tiny по памяти. Для него сохраняйте **`--img-size 224`**: предобученная модель ожидает сетку патчей для этого разрешения. При нехватке GPU-памяти уменьшайте `--batch-size`.

Официальная карточка весов: https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k

**Проверяем, помогает ли ViT ансамблю на fold 0:**

```powershell
.\.venv\Scripts\python.exe check_validation.py --oof checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0_oof.csv checkpoints/vit_base_patch16_224_augreg_in21k_ft_in1k_fold0_oof.csv
```

Эта команда покажет общую `macro-F1` на том же наборе картинок и список ошибок `outputs/validation_errors.csv`. Если ансамбль хуже, чем ConvNeXt в одиночку, не добавляйте ViT без дальнейшей проверки/подбора весов.

## 8. Создаём submission для Kaggle

Одна ConvNeXt:

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoints checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0.pt --batch-size 16
```

Три фолда ConvNeXt в ансамбле:

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoints checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0.pt checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold1.pt checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold2.pt --batch-size 16
```

Ансамбль ConvNeXt и ViT (если улучшил проверку качества):

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoints checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0.pt checkpoints/vit_base_patch16_224_augreg_in21k_ft_in1k_fold0.pt --batch-size 8
```

По умолчанию работает **TTA=2** (оригинал + зеркальное отражение), усредняются вероятности. Можно отключить TTA через `--tta 1` и сравнить результат на Kaggle / отдельной валидации. Для **10 точных train/test дубликатов** результат заменяется известной меткой автоматически (отключить можно `--no-exact-match`).

Результаты:

```text
outputs/submission.csv                ← ЭТО загружаем на Kaggle
outputs/submission_probabilities.csv  ← вероятности, уверенность, имена классов
```

`submission.csv` будет содержать ровно **849 строк** в порядке исходного шаблона:

```csv
id,label
c41628b1-4781-4392-ac8d-6bfe981f73f9,9
...
```

Число `9` выше — **пример формата, не реальное предсказание для этого ID**.

На странице Kaggle откройте соревнование → `Submit Predictions` → выберите **`outputs/submission.csv`** → отправьте. Kaggle выдаст публичный leaderboard score, после чего можно оценить варианты моделей.

## 9. Дополнительные совпадения для повышения точности (по желанию)

Скрипт подготовки уже создал `data/near_duplicate_candidates.csv` — список близких изображений. Часть из них не идентична, поэтому **не следует без проверки принудительно подставлять их метки**.

Создайте визуальную страницу:

```powershell
.\.venv\Scripts\python.exe review_matches.py
```

Откройте двойным кликом `outputs/review_duplicates.html`. Там по две картинки рядом: TEST и ближайшее TRAIN, а также предложенная метка. Отметьте галочками **только очевидно правильные совпадения по персонажу** и нажмите `Скачать подтверждённые совпадения CSV`. Скачанный файл `manual_overrides.csv` переместите в папку `outputs`.

После этого пересчитайте предсказания с ключом `--overrides`:

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoints checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0.pt checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold1.pt checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold2.pt --overrides outputs/manual_overrides.csv --batch-size 16
```

Итоговый CSV будет содержать подтверждённые вручную метки. Если хотите только автоматический результат, `--overrides` не нужен.

## 10. Своя сеть меньше 5 МБ

`small_model.py` содержит оригинальную `SmallAnimeCNN`:

- первый обычный свёрточный блок извлекает низкоуровневые визуальные признаки;
- **inverted residual blocks** расширяют число каналов и применяют *depthwise convolution* для экономии параметров;
- **Squeeze-and-Excitation** перераспределяет важность каналов;
- остаточные соединения в блоках одного размера помогают обучению;
- global average pooling превращает карту признаков в вектор;
- линейный слой отображает 256 признаков в 18 классов.

Размер сохранённого тестового checkpoint **1 971 875 байт** при 469 050 обучаемых параметрах. Это собственная CNN **с нуля**, а не готовые веса/файнтюнинг. `small_model.py` можно показать преподавателю вместе с объяснением выше.

Обучение без предобученных весов:

```powershell
.\.venv\Scripts\python.exe train_small.py --fold 0 --epochs 90 --batch-size 64 --workers 2
```

После обучения появляется `checkpoints/custom_tiny_cnn_fold0.pt`. На каждой новой лучшей эпохе проверяется, что **размер файла меньше 5 000 000 байт**.

Сделать отдельный submission этой модели:

```powershell
.\.venv\Scripts\python.exe predict.py --checkpoints checkpoints/custom_tiny_cnn_fold0.pt --output outputs/submission_small.csv
```

**Ожидание:** небольшая модель, обученная с нуля на 2915 картинках, может уступать предобученной ConvNeXt; этого нельзя исключать. Не заменяйте основной submission маленькой моделью, пока не сравните F1.

### Опционально: distillation

Если fold 0 ConvNeXt уже обучен, можно попробовать обучить маленькую модель, кроме настоящих меток ориентируясь на вероятности сильной модели:

```powershell
.\.venv\Scripts\python.exe train_small.py --fold 0 --epochs 90 --batch-size 32 --workers 2 --teacher-checkpoint checkpoints/convnext_tiny_fb_in22k_ft_in1k_fold0.pt
```

Тогда студент **по-прежнему использует собственную архитектуру и начинает со случайных весов**, но получает дополнительный сигнал от учителя. В коде реализована дистилляция softmax с температурой 2.5: `65% CE` + `35% KL`. Проверьте с преподавателем, признаётся ли такая форма обучения соответствующей условию дополнительных баллов.

Не сравнивайте сохранённый `custom_tiny_cnn_fold0.pt` после дистилляции с прежней версией: имя будет одинаковым и файл перезапишется. Если нужно сохранить оба варианта, используйте отдельные папки через `--save-dir checkpoints/distilled`.

## 11. Техническое описание ключевых файлов

| Файл | Что делает | Что можно настраивать |
|---|---|---|
| `prepare_data.py` | Распаковывает все ZIP и создаёт `fold` | `--n-folds`, `--seed` |
| `common.py` | Читает и нормализует изображения, делает аугментации | `make_transforms`: `RandomResizedCrop`, flip, color jitter, RandomErasing |
| `train.py` | Дообучает CNN или ViT, сохраняет лучший checkpoint по validation `macro-F1` | `--model`, `--fold`, `--epochs`, `--batch-size`, learning rates, `--weighting` |
| `predict.py` | Усредняет вероятности нескольких checkpoint, TTA и точные дубли | `--checkpoints`, `--tta`, `--overrides` |
| `check_validation.py` | Сравнивает несколько моделей на **тех же** валидационных ID | `--oof` |
| `small_model.py` | Объявляет собственную CNN, не загружает готовые веса | архитектуру и число каналов |
| `train_small.py` | Обучает собственную CNN и следит за размером файла | `--teacher-checkpoint`, `--lr`, `--epochs` |
| `review_matches.py` | Создаёт HTML, чтобы вручную проверить похожие test/train | `--output` |

### Как читать F1

В многоклассовой классификации `macro-F1` — среднее F1 по 18 персонажам без учёта числа картинок в классе. Это важно: `Luffy` представлен заметно реже, чем `Nami`. `weighted-F1` усредняет с весами по числу изображений, а `accuracy` — обычная доля правильных классификаций. Соревнование лишь сообщает `F1`, но не раскрывает в предоставленном тексте способ усреднения: фактическую метрику leaderboard нужно проверять в Kaggle.

### Если в VS Code возникает ошибка

| Проблема | Что проверить |
|---|---|
| `No module named 'timm'` | Установите `requirements.txt` **в активной `.venv`** |
| `torch.cuda.is_available() == False` | Установлен ли CUDA-PyTorch, NVIDIA-драйвер и правильный интерпретатор |
| `CUDA out of memory` | Уменьшите `--batch-size`; для ViT используйте `8`, `4`, `2` |
| `FileNotFoundError raw/...` | Шесть файлов лежат непосредственно в папке `raw/`, не уровнем ниже |
| `DataLoader worker exited unexpectedly` | Используйте `--workers 0` |
| Не скачиваются готовые веса | Проверьте доступность Hugging Face и интернет, это не проблема исходных ZIP |
| Kaggle отклоняет CSV | Используйте **новый** `outputs/submission.csv`, а не шаблон `raw/submission.csv` |
| У F1 появляются перепады | Повторите эксперимент с тем же `fold` и `seed`, проверьте конфликты меток/похожие картинки и confusion/errors |

## 12. Порядок экспериментов для хорошего результата

1. `prepare_data.py` → подготовить датасет.
2. `train.py` ConvNeXt fold 0 → получить первый validation F1.
3. ConvNeXt folds 1–2 → получить ансамбль и первый настоящий Kaggle submission.
4. `review_matches.py` → подтвердить явные почти-дубли и сделать обновлённый submission.
5. ViT fold 0 → проверить, добавляет ли он качества на **том же** validation fold.
6. При необходимости обучить ConvNeXt folds 3–4, сравнить публичный score.
7. `train_small.py` → отдельная компактная модель ради дополнительных баллов, документировать её архитектуру и размер.

**Важно:** лучше сохранять таблицу `модель | fold | seed | epoch | macro-F1 | weighted-F1 | Kaggle score` и изменять по одному параметру. Не делайте вывод по одному удачному публичному leaderboard score: публичная часть теста может быть небольшой, поэтому полезно ориентироваться и на честную валидацию.
