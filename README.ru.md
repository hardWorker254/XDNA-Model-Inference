# NpuEmbeddings — Rust-рантайм

Инференс BERT-семейства (эмбеддинги), а также STT, классификация, поза/руки и OCR на AMD NPU (XDNA1/npu1, XDNA2/npu2) с резервным путём на CPU и опциональным Vulkan-бэкендом.

Это дерево — **переписывание рантайма на Rust**. C++-рантайм остаётся в соседнем дереве как **оракул паритета**.

---

## Два дерева (один репозиторий, worktree)

| | это дерево | оракул |
|---|---|---|
| путь | `~/XDNA-Model-Inference` | `~/Npu-Embeddings-XDNA1` |
| ветка | `rust/clean-tree` | `main` |
| рантайм | Rust | C++ |
| роль | разрабатываемый код | эталон для сверки |

Полная историческая документация C++-дерева (большой README, сборка, точность) — в оракуле. Этот README описывает текущее состояние Rust-дерева.

---

## Структура

```
backends/cpu         host-kernels (эталон чисел)
backends/vulkan      GPU-бэкенд (WGSL/ash)
backends/xdna        NPU: транспорт, диспетчеризация, геометрия
core/npum-format     контейнер .npum/.npue (чтение)
core/npum-model      тензоры, граф, шов арифметики (Ops/Backend)
core/npum-tokenizers токенизаторы (WordPiece, Unigram, Gemma, Whisper-BPE, XLM-R)
core/npum-families   семейства, каталог, отказы
cli/                 бинарники npuembeddings, npuaudio, npuimage
server/              HTTP-сервер (axum)
tools/               Python-инструменты (упаковка, parity, verify)
docs/                документация (en/ru)
```

Правило зависимостей: `core/` не знает о `backends/`, `backends/` не знают о `cli/`/`server/`.

## Карта контуров репозитория

| Путь | Назначение | API/стабильность | Основные потребители | Примечания |
|---|---|---|---|---|
| `core/npum-format` | Контейнер `.npum/.npue` (чтение/запись, дайджесты, provenance) | Канонический источник истины по формату; Rust API в workspace (`rlib` + `cdylib`) | `core/*`, `backends/*`, `cli`, `server`, `tools`/Python через FFI | Единственный writer формата; reader принимает legacy `.npue` и `.npum` |
| `core/npum-model` | Тензоры, типы модели/графа, швы `Backend`/`Ops`, caps/dispatch | Контракты core-слоя (внутренние, стабильность по соглашению workspace) | `backends/*`, `core/npum-families`, `cli`, `server` | `Backend` — tensor-ориентированный контракт; `Ops` — slice-шов для семейств/forward-pass |
| `core/npum-tokenizers`, `core/npum-families` | Токенизация и правила семейств/каталога/владения | Контракты core-слоя | `cli`, `server`, backend/семейные реализации | Владение семействами проверяется на границах маршрутизации/CLI |
| `backends/cpu`, `backends/vulkan`, `backends/xdna` | Провайдеры вычислений и device-специфичные пути | Реализации backend’ов, выбор на краю | `cli`, `server`, тесты/инструменты | CPU — эталон чисел; Vulkan/XDNA имеют ограничения возможностей/точности |
| `cli/` | Пользовательские бинарники (`npuembeddings`, `npuaudio`, `npuimage`) | CLI-контракт (флаги/вывод/отказы) | Пользователи, скрипты, parity-harness | Composition root выбора backend/семейства |
| `server/` | HTTP-сервис и обработчики маршрутов | HTTP-контракт | Клиенты/сервисы, пути `serve` | Единый набор маршрутов; режим определяется контейнером/семейством |
| `tools/` | Build-time/export/pack/verify/gen/research/data инструменты | Внутренний tooling (эволюционирует вместе с пайплайном) | Разработка, CI/parity/build-процессы | В основном Python; детали в [`tools/README.md`](tools/README.md) |
| `reference/` | Оракул/эталонная среда и входы для сравнения | Не runtime-API; только parity/reference | Запись/сравнение parity, валидация | Не продуктовый runtime-код; отделён от Rust-runtime |
| `kernels/` | Исходники AIE-ядер для экспорта/сборки NPU-дизайнов | Источники артефактов (не Rust API) | Пайплайн `tools/export`, артефакты для `backends/xdna` | См. [`kernels/README.md`](kernels/README.md) |
| `python/` | Удобные Python-обёртки/фасады поверх runtime-потоков | Удобный слой, не каноническая runtime-логика | Пользователи/скрипты на Python | Сейчас небольшой; делегирует в CLI/server/runtime |
| `npue_tools/` | Область Python-пакета/ассетов в переходном состоянии | Переходный/внутренний (минимальный) | Инструменты/процессы, которые ещё на него ссылаются | Не канонический writer формата и не runtime-реализация |
| `docs/` | Контракты и документация (usage/build/backends/models/parity) | Документация (нормативна где указано, напр. инварианты/parity) | Разработчики/пользователи/ревьюеры | EN/RU для нормативных изменений должны быть согласованы |
| `models/`, `weights/` (локальные layout’ы) | Хранилище контейнеров/моделей и пути разрешения | Соглашение о данных/расположении, не API кода | Разрешение моделей в CLI/server и tooling | Содержимое может быть локальным; пути определяются во время работы |

---

## Сборка

```sh
./bootstrap.sh          # Python venv с зависимостями для tools
cargo build --release
```

Бинарники: `target/release/{npuembeddings,npuaudio,npuimage}`.

Проверка отсутствия XRT/Python в direct-сборке (гейт): `python3 tools/verify/verify_no_xrt.py` (в контексте дерева, где доступен скрипт).

---

## Использование (кратко)

- `npuembeddings list|add|tokenize|embed|serve`
- `npuaudio transcribe|serve`
- `npuimage classify|pose|hands|mppose|ocr|serve`

Режим/эндпоинт сервера определяется `arch` контейнера, а не флагом. Модель можно передать путём к `.npue/.npum` или именем (поиск в `models/`, а также в `weights/<family>/<model>/`).

### Выбор бэкенда/путей
- `--backend cpu|vulkan` — выбор арифметического бэкенда (где поддерживается).
- `--vk-ops gemm,layn,gelu,softm` — выборочный offload на Vulkan (подразумевает Vulkan); `--backend vulkan` без `--vk-ops` включает поддерживаемые ops.
- `--npu-ops ...` — массивный/NPU-путь операций (где реализован), независим от `--backend`; неподдерживаемые комбинации/семейства отказывают по имени.

Embed-сервер поддерживает Vulkan ops; прочие серверные движки в текущем состоянии работают на хосте.

---

## Контейнер и модели

- Формат: `.npum` (принимается также legacy `.npue`).
- Источник весов для упаковки — safetensors (где применимо), ONNX остаётся для графов/vision-случаев.
- Упаковка: Python (`tools/pack/pack_npue.py`).

---

## Паритет и инварианты

- Инварианты: [docs/en/INVARIANTS.md](docs/en/INVARIANTS.md), [docs/ru/INVARIANTS.md](docs/ru/INVARIANTS.md).
- Паритет с C++-оракулом: `python3 tools/parity/run.py record|compare` (golden’ы в `tools/parity/golden/` — контракт; осмысленные расхождения фиксируются в `tools/parity/divergences.txt`).

Дополнительно: [docs/ru/BUILDING.md](docs/ru/BUILDING.md), [docs/ru/USAGE.md](docs/ru/USAGE.md), [docs/ru/BACKENDS.md](docs/ru/BACKENDS.md).

Лицензия: Apache-2.0 (`LICENSE`, `THIRD-PARTY.md`).
