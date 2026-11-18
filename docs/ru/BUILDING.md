# Сборка (Rust-дерево)

Документ описывает сборку и базовую валидацию Rust-рантайма в этом worktree.

## Поддерживаемые сценарии

| Сценарий | Что нужно | Что получаете |
|---|---|---|
| **Только Rust** | Rust toolchain (см. `rust-toolchain.toml`), `cargo` | `cargo build --release` для host-side крейтов/бинарников (с учётом доступных фич/зависимостей окружения) |
| **Rust + Python tooling** | Python 3, venv + зафиксированные зависимости | Упаковка, reference/verify/parity-скрипты в `tools/`, `reference/`, `npue_tools/` |
| **Полное окружение NPU/export/parity** | Всё выше + драйвер/железо AMD XDNA, design-артефакты и (для части export-потоков) vendor wheels, совместимые с матрицей bootstrap | NPU-пути выполнения, design-aware parity, export/packaging, зависящие от vendor runtime |

Используйте минимальный сценарий под задачу. Тяжёлые проверки (модели/NPU/полный parity) не обязательны для каждого Rust-правления, но обязательны при изменении поведения/контрактов на соответствующих путях.

## Требования

- Rust toolchain зафиксирован в [`rust-toolchain.toml`](../../rust-toolchain.toml) (`1.85`, компоненты `rustfmt`, `clippy`).
- Python 3 для tooling (`tools/`, `reference/`). `bootstrap.sh` сейчас ориентирован на фиксированную матрицу AMD/vendor wheels (в т.ч. Python **3.14 / cp314** для отдельных пакетов) и проверяет контрольные суммы там, где они заданы.
- Для NPU: поддерживаемое железо/драйвер AMD XDNA1/XDNA2 и совместимые design-артефакты (несовместимость должна приводить к отказу, fail-closed).
- Для Vulkan backend: рабочий Vulkan ICD/устройство.

## Bootstrap (Python tooling)

```sh
./bootstrap.sh
```

`bootstrap.sh` создаёт локальный venv и ставит зафиксированные зависимости/колёса для поддерживаемого tooling/export-окружения. Если интерпретатор/платформа не соответствуют матрице, bootstrap завершится с ошибкой; в этом случае используйте Rust-only поток или подготовьте совместимое окружение.

## Сборка

```sh
cargo build --release
```

Артефакты:
- `target/release/npuembeddings`
- `target/release/npuaudio`
- `target/release/npuimage`

## Форматирование и линт

```sh
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

Базовая дисциплина форматирования/линта ужесточается; прогоняйте проверки для затрагиваемых изменений и исправляйте регрессии, внесённые вашими правками. Избегайте переформатирования несвязанных файлов в том же изменении без явной цели нормализации дерева.

## Валидация

- **Rust-тесты:** `cargo test --workspace` (или `--release` при необходимости).
- **Паритет с C++-оракулом:** `python3 tools/parity/run.py compare --candidate ./target/release` (нужны записанные golden'ы и установленные модели/fixtures для нескипнутых кейсов). См. [PARITY.md](PARITY.md).
- **No-XRT gate:** `python3 tools/verify/verify_no_xrt.py` (когда скрипт и контекст сборки доступны).

## Репозитории / worktree

Это дерево может быть `git worktree` оракул-репозитория. Бинарники оракула, планы и часть tools/docs могут находиться в соседнем дереве; для сравнения используйте абсолютные пути к артефактам оракула.
