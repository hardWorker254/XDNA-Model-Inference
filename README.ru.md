# NpuEmbeddings

NpuEmbeddings — это Rust-workspace для инференса моделей на AMD NPU (XDNA1/npu1, XDNA2/npu2) с эталонным CPU-путём и опциональным ускорением на Vulkan для поддерживаемых операций.

Проект включает три CLI-бинарника, общий HTTP-сервер, инструменты для работы с контейнерами/моделями и документацию по сборке, использованию, бэкендам и проверкам.

## Бинарники

| Бинарник | Основные команды | Назначение |
| --- | --- | --- |
| `npuembeddings` | `list`, `add`, `tokenize`, `embed`, `serve` | Сценарии текстовых эмбеддингов |
| `npuaudio` | `transcribe`, `serve` | Распознавание речи (speech-to-text) |
| `npuimage` | `classify`, `pose`, `hands`, `mppose`, `ocr`, `serve` | Сценарии компьютерного зрения |

## Структура workspace

```text
backends/cpu/         CPU-ядра, арифметика и эталонные пути выполнения
backends/vulkan/      Vulkan-бэкенд
backends/xdna/        Транспорт/диспетчеризация NPU и device-специфичное выполнение
core/npum-format/     Формат контейнера (.npum/.npue): чтение/запись, дайджесты, метаданные
core/npum-model/      Тензоры, runtime-абстракции, контракты backend/ops
core/npum-tokenizers/ Токенизаторы
core/npum-families/   Семейства моделей, каталог и правила маршрутизации/валидации
cli/                  CLI-бинарники и диспетчеризация команд
server/               HTTP-сервер и маршруты
tools/                Упаковка, экспорт, проверки, генерация и вспомогательные скрипты
docs/                 Документация для пользователей и разработчиков (EN/RU)
kernels/              Исходники AIE/NPU-ядер для пайплайна экспорта/сборки
models/               Локальные метаданные/контейнеры моделей (крупные веса обычно хранятся локально)
```

Направление зависимостей (по дизайну): `core/` не зависит от `backends/`, а `backends/` не зависят от `cli/` и `server/`. Выбор бэкенда и семейства выполняется на краях системы (CLI/server).

## Сборка

Набор зависимостей зависит от используемого пути (только CPU, Vulkan, NPU/export toolchain). Для Python-окружения инструментов из `tools/` в репозитории есть bootstrap-скрипт.

```bash
./bootstrap.sh
cargo build --release
```

Собранные бинарники будут в `target/release/`:

- `npuembeddings`
- `npuaudio`
- `npuimage`

## Использование (кратко)

- Модель/контейнер можно передать путём (`.npum` или legacy `.npue`) или, где поддерживается, по имени (разрешение в настроенных расположениях моделей).
- Режим и набор HTTP-эндпоинтов определяются семейством/`arch` контейнера (а не отдельным флагом режима).
- Параметры выбора бэкенда и маршрутизации операций (CPU/Vulkan/NPU ops) зависят от команды, семейства и доступных артефактов/возможностей.

Подробности по командам и флагам: `--help` у бинарников/подкоманд и документация в `docs/`.

## Документация

- [docs/en/BUILDING.md](docs/en/BUILDING.md) / [docs/ru/BUILDING.md](docs/ru/BUILDING.md) — сборка и проверки
- [docs/en/USAGE.md](docs/en/USAGE.md) / [docs/ru/USAGE.md](docs/ru/USAGE.md) — использование CLI/server и разрешение моделей
- [docs/en/BACKENDS.md](docs/en/BACKENDS.md) / [docs/ru/BACKENDS.md](docs/ru/BACKENDS.md) — выбор бэкендов и возможности
- [docs/en/MODELS.md](docs/en/MODELS.md) / [docs/ru/MODELS.md](docs/ru/MODELS.md) — контейнеры, модели и расположение весов
- [docs/en/PARITY.md](docs/en/PARITY.md) / [docs/ru/PARITY.md](docs/ru/PARITY.md) — проверки/паритет
- [docs/en/INVARIANTS.md](docs/en/INVARIANTS.md) / [docs/ru/INVARIANTS.md](docs/ru/INVARIANTS.md) — инварианты и контракты runtime
- [tools/README.md](tools/README.md) — обзор инструментов

## Формат контейнера

Основной формат контейнера — `.npum` (NPU Model). Для совместимости принимаются legacy-контейнеры `.npue`; при записи используется `.npum`.

## Лицензия

Apache-2.0. См. [LICENSE](LICENSE), [THIRD-PARTY.md](THIRD-PARTY.md) и [PROVENANCE.md](PROVENANCE.md).
