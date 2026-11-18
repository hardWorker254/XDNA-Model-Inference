# Использование

## Бинарники и семейства
- `npuembeddings` — текстовые эмбеддинги: `list`, `add`, `tokenize`, `embed`, `serve` (`/v1/embeddings`).
- `npuaudio` — распознавание речи (Whisper): `transcribe`, `serve` (`/v1/audio/transcriptions`).
- `npuimage` — зрение: `classify`, `pose`, `hands`, `mppose`, `ocr`, `serve` (`/v1/classify`, `/v1/pose` и т.д. по модели/`arch`).

Эндпоинт/режим сервера выбирается полем `arch` контейнера, а не флагом. Контейнер чужого семейства/бинарника отклоняется по имени.

## Разрешение модели
Аргумент модели может быть:
- путём к контейнеру (`.npue`/`.npum` или путь с разделителями) — используется как есть;
- именем модели — ищется в `<root>/models/<name>.npue|.npum`, затем в `weights/<family>/<name>/` (например `weights/<family>/<name>/model.npue` или `<name>.npue`).

`<root>` — корень рантайма (см. `--help`).

## Основные флаги
- `--backend cpu|vulkan` — арифметический бэкенд (где поддерживается).
- `--vk-ops gemm,layn,gelu,softm` — выборочный offload на Vulkan (подразумевает Vulkan). Неподдерживаемые семейства/пути отказывают по имени.
- `--npu-ops ...` — NPU/массивный путь операций (где реализован). `--npu-ops` и `--backend` — разные оси (массивный путь vs арифметический бэкенд).

Правило парсинга: побеждает последнее вхождение флага.

## Примеры
```sh
./target/release/npuembeddings embed models/all-MiniLM-L6-v2.npue in.txt out.f32
./target/release/npuembeddings serve models/all-MiniLM-L6-v2.npue --port 8080 --bind 127.0.0.1
./target/release/npuembeddings serve models/all-MiniLM-L6-v2.npue --vk-ops gemm
./target/release/npuimage pose models/yolov8n-pose.npue image.jpg --npu-ops conv
```

(Точные флаги и доступность зависят от семейства и сборки; неподдерживаемые комбинации завершаются отказом по имени, а не тихим fallback.)
