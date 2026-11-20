# План: включение XDNA для всех семей моделей (CLI и `serve`)

SPDX-License-Identifier: Apache-2.0

## 0. Главное уточнение архитектуры

В `backends/xdna` есть **два слоя**, и только один из них считает на массиве:

1. `npum_xdna::XdnaBackend` (`backends/xdna/src/lib.rs:77`) реализует `npum_model::Backend`, но **все** его методы (`dispatch.rs:49-135`) возвращают `Error::NotImplemented`. Это устаревший слой; `prepare()` при этом объявляет `OpKind::ALL` (`lib.rs:119-134`) — использовать это как признак готовности **нельзя**.
2. Реальный путь — сейм-трейты `core/npum-model/src/array.rs`: `ArrayGemm` (`:224`), `ArrayAttention` (`:422`), `ArrayText` (`:490`), `ArrayConvs` (`:325`), `ArrayEltwise` (`:145`). Их реализуют `ArrayGemms` (`backends/xdna/src/array_gemms.rs:196`), `ArrayConvs` (`array_convs.rs:73`), `Eltwise` (`eltwise.rs:425`) и `attention::run` (`attention.rs:410`). Диспатч идёт только через `ArrayDesign` (`array.rs:90`), который один выделяет/биндит/запускает буферы.

**Вывод:** новые подключения делать через `array`-трейты, а не через `Backend`/`dispatch`. `--npu-ops` — это вопрос «где выполняется операция»: пока сейма нет, код должен **отвергаться по имени**, а не молча считаться на хосте (`cli/src/families.rs`). Отказ в `cli/src/serve.rs` теперь применяется ко ВСЕМ семействам, включая Embed (раньше Embed его обходил и молча запускал хост при `--npu-ops`); списки заменены реальными: Embed/Stt — `gemm,attn,layn,gelu,softm`, Cls — те же плюс `conv`, Pose — `conv`. Hands/MpPose/OCR остаются с пустым списком, пока у них нет HTTP-движка.

## 1. Текущее состояние

| Семейство | CLI | `serve` | Дизайн-сеты | Что не хватает |
|---|---|---|---|---|
| Embed (BERT) | Да: `gemm/layn/gelu/softm/attn` | Да (этап G): `Embedder::with_xdna`, тот же список кодов | есть | — |
| GTE | Да (этап E): `gemm/attn/layn/softm` (без `gelu` — FFN гейтированный) | Да (этап G): общий с BERT путь `Embedder::with_xdna` | есть: экспортирован `gte-multilingual-base/artifacts_npu1`, упакован в `.npue` | — |
| EmbeddingGemma | Да (этап F): `gemm/softm`, остальные коды отвергаются по имени | Да (этап F): `EmbedModel::Gemma`, тот же список | есть: `embeddinggemma-300m/artifacts_npu1`, упакован в `.npue` |Четыре RMSNorm и GeGLU — на хосте (нет дизайнов); MQA-attention считается на хосте, но QK/V — через `ArrayText` |
| Whisper STT | Да (этапы B–D + G + F): `gemm/attn/fft/mproj/layn/gelu/softm/logit`, два набора `gemm_rtp` + `gemm_rtp_dec` | Да (этапы D+G+F), `logit` через `ArrayLogits` в `XdnaForStt` | есть: `gemm_rtp_dec` экспортирован (35 потоков), контейнер упакован | decoder `cross_kv` и conv1/conv2 считаются на хосте; GELU `erf` хоста против полинома дизайна |
| ViT Cls | Да (этапы B–D): `gemm/attn/layn/gelu/softm/conv`; `conv` = patch-embed GEMM, `gemm` = 4 GEMM на слой | Да (этап D) | есть (`vit-base-patch16-224/...`) | head (1000) не кратен `tile_n*cols=192`; не-свёрточные узлы — на хосте. Хостовый path с bf16-контейнером починен общим `operand::read_operand` (де-tiling панелей) |
| Pose | `conv` через `ArrayConvs` | Да (этап G): `PoseDetector::with_xdna`, список `conv` | есть/частично | depthwise conv и не-свёрточные узлы остаются на хосте |
| Hands / MpPose / OCR | `conv` через `ArrayConvs` | нет HTTP-движков, отказ по имени | есть/частично | XDNA-инъекция в движки (движков нет); MpPose/OCR не имеют endpoint-а в эталоне |

## 2. Этапы реализации

> Статус: **этапы A, B, C, D и E реализованы; этап G — каркасы tests добавлены,
> полная проверка требует платы и отсутствующих дизайн-сетов**.
> Финальная доделка: `serve` больше не обходит отказ `--npu-ops` для Embed и
> инъецирует XDNA-хендлы в `Embedder` (BERT и GTE, тот же сейм, что у CLI);
> `PoseDetector` получил `with_xdna` и диспатчит свёртки через `ArrayConvs`,
> так что `conv` в списке serve для Pose теперь честный. Оставшиеся семейства
> (Hands/MpPose/OCR) не имеют HTTP-движков и продолжают отказываться по имени.
> Gemma (этап F): `backends/cpu/src/pipelines/gemma.rs` — отдельный forward для `gemma3_mqa_rope_geglu` (MQA 3:1, RoPE с двумя базами, четыре RMSNorm, GeGLU tanh, mean-pooling + две Dense + L2). Важный момент: контейнер хранит СЫРОЙ вес RMSNorm, а эталонный ONNX — уже `1+w`, поэтому `+1` складывается при загрузке весов, а не в ядре. Дизайны `gemm_rtp` и `softmax/` дают честные коды `gemm` и `softm`; `layn`, `gelu`, `attn` отвергаются по имени. Паритет на плате: 1−cos = 4.7e-3 / 3.1e-3 (допуск 1e-2), `cli/tests/gemma_array_parity.rs`; CLI и `/v1/embeddings` через `serve` работают с `--npu-ops gemm,softm`.
> Whisper `logit` (этап F): восемь потоков `logits_0..7` в `gemm_rtp_dec` — это
> не 8 batch-тиров, а 8 срезов словаря по 6656 колонок (51865 = 7×6656 + 5273).
> `ArrayGemms::open` при `--npu-ops logit` транспонирует `decoder.embed_tokens`
> в восемь bf16-панелей [512,6656] и тилирует их через `stage_frontend_b`;
> `ArrayLogits::logits` диспатчит все восемь и собирает плоский `[vocab]`.
> Хостовый `logits_from_into` остался побайтно прежним, а `logit` теперь в
> честных списках CLI и `serve`. На плате: 1−cos = 1.7e-6, argmax совпадает,
> 8 диспатчей на шаг; `--npu-ops logit` даёт тот же транскрипт ` you`.
> G — device-gated (`cli/tests/text_array_parity.rs`) прогоняет BERT и GTE через
> `ArrayText`/`ArrayEltwise` и сравнивает 1−cos и счётчик диспатчей; пропускается
> без `/dev/accel/accel0` и без контейнера с панелями (переменные
> `NPUEMBEDDINGS_{BERT,GTE}_NPU`, `NPU_ARTIFACTS`). Замер на bge-base: 1−cos =
> 1.05e-3 при 1237 диспатчах; GTE 2.27e-3 при 1225; допуск 5e-3. ViT на
> `bus.jpg` — 3.06e-5 при 374 диспатчах, допуск 1e-3. Whisper (`gemm,attn`,
> оба набора) — транскрипт совпал, dispatching подтверждён счётчиками бэкендов
> (17340 в CLI). Счётчики читаются из бэкендов, а не из флагов. Device-independent
> `npuembeddings serve` с `--npu-ops gemm,attn,layn,gelu,softm` на BERT и платой
> тоже проверен: `/v1/embeddings` вернул настоящие вектора, статус показал 4
> hw_context'а и все пять кодов на массиве.
> `gte::gelu_numerics_tests` сравнивает полином `kernels/gelu_poly.cc` с точным
> `erf` хоста на сетке [−5, 5] (2.494e-3, предел 3e-3) — это же относится к
> Whisper `stt/gelu`. `cli/tests/pipeline_array_parity.rs` добавляет каркасы
> для ViT (1−cos logits, допуск 1e-3) и Whisper (диспатч + число окон; текст
> дискретен, поэтому проверяется диспатч и число окон). Замеры допусков выполнены на этой плате (см. выше).
> Дизайн GTE экспортирован (`runtime/artifacts/gte-multilingual-base/artifacts_npu1`,
> один xclbin, 18 потоков) и упакован в `models/gte-multilingual-base.npue`.
> У GTE гейтированный FFN, поэтому отдельного `gelu/` дизайна нет и код
> `gelu` для этой архитектуры отвергается по имени.
> При реальном прогоне нашёлся и исправлен баг: `Vit::from_container` читал
> BF16-панели как F32, из-за чего `frontend.patch_embed` имел половину нужной
> длины и хостовый GEMM падал. Чтение вынесено в общий
> `backends/cpu/src/pipelines/operand.rs` (де-tiling панелей); CLI снова
> классифицирует `bus.jpg` как `minibus`.
> Экспорт на этой машине требует двух окружений (без них падает на последнем
> шаге или не находит `pyxrt`):
> `PATH=/opt/xilinx/xrt/bin:$PATH PYTHONPATH=/opt/xilinx/xrt/python ./.venv/bin/python tools/export/export_gemm_rtp.py --arch 1 --target <model>`.
> Во время компиляции дизайна плата занята: параллельные device-gated тесты
> получают `DRM_IOCTL_AMDXDNA_CREATE_HWCTX ... -22` и должны запускаться после.
> Экспортированы GTE (`gemm_rtp`, 18 потоков + layernorm/softmax) и Whisper
> (`gemm_rtp_dec`, 35 потоков), упакованы `gte-multilingual-base.npue` и
> `whisper-base.npue`.
>
> Реальные прогоны вскрыли и закрыли четыре дефекта, которых mock-тесты не
> видели: (1) ViT и Whisper читали BF16-панели как F32 и падали на укороченном
> операнде — теперь общий `pipelines/operand.rs` с де-tiling; (2) `ArrayGemms`
> отказывал на строк больше tiers — добавлен по-тилевый split GEMM (1500 → 3×512
> для Whisper); (3) внимание отказывало, когда `seq > tier_rows` (`tier_batch`
> был нулём) — теперь `max(batch, tier_rows/seq)`; (4) `ensure_run` обрезал
> `a_q` по `seq`, хотя буфер держит только один tier, — индекс зажат. Плюс
> `Transcript::n_dispatches` перестал быть заглушкой и читается из бэкендов.
>
> Stage F (фронтэнд Whisper) закрыт для `fft` и `mproj`: потоки `dft400` и
> `mel_proj` уже были в экспортированном `gemm_rtp`, но их B-операнды НЕ
> вшиты в xclbin — бэкенд сам генерирует матрицу twiddle и Slaney-банк,
> тайлит их в `b_layout` дизайна и стейджит как обычный panel (`pipelines/
> operand`-подход для GEMM, `stage_frontend_b` для фронтэнда). Добавлен
> `ArrayFrontend` (`dft400`/`mproj`) в `core/npum-model/src/array.rs`; он
> супертрейт `ArrayText`, поэтому один хэндл энкодера обслуживает и GEMM, и
> фронтэнд. Эпилог `log10` после mel-проекции теперь общий для хоста и массива
> (без него транскрипт превращался в «and and and»). Замер на плате: DFT
> max rel err 1.76e-2 (bf16 twiddles, как и обещал экспортер), mel 2.24e-3;
> транскрипт `fft,mproj` совпадает с хостом, dispatching подтверждён.
> `npuaudio serve --npu-ops fft,mproj` тоже проверен по HTTP: `/v1/audio/transcriptions` на `tone.wav` вернул тот же транскрипт, что и хост. Полный `cargo test --workspace` с платой: exit 0, 51 `test result: ok`.
> A — добавлен `cli/src/npu.rs`: общий поиск дизайнов (`--artifacts` учитывается
> и при открытии, и в статус-блоке), открытие GEMM/eltwise/conv, проверки
> BF16-панелей и `b_layout_hash`, сборка спеков из геометрии (`TextGeometry`:
> BERT/GTE); `embed.rs`, `arrayconv.rs`, `status.rs`, `serve.rs`, `vit.rs`,
> `stt.rs` переведены на него.
> B — сеймы в CPU-пайплайнах: `whisper.rs` (энкодер, декодер, внимание, LN/GELU/
> softmax), `vit.rs` (patch embed, 4 GEMM на слой, внимание, LN/GELU), сквозная
> передача через `whisper_session.rs`. При `None` все вызовы побайтно прежние;
> добавлены проверки ширины score-строки и mock-тесты диспетчеризации.
> C — CLI-глаголы открывают дизайны и передают сеймы: `stt.rs` — два набора
> (`gemm_rtp` + `gemm_rtp_dec`) и eltwise, `vit.rs` — один набор, причём
> `conv` диспатчит patch-embed GEMM, а `gemm` — четыре GEMM слоя (эксплицитные
> флаги, потому что `gemm_on_array()` истинён и для одного patch-панеля).
> Отказы `refuse_array_not_ported(..., &[])` заменены реальными списками;
> статус-блок печатается до отказов и называет array только реально
> запрошенные проходы с наличным дизайном; счётчик диспатчей берётся из
> самих бэкендов. `check_context_budget` считает контексты по флагам.
> D — `serve` открывает дизайны один раз при старте и инъецирует хендлы в
> движки: `Transcriber`/`Classifier` хранят их за отдельными `Mutex` (транспорт
> `Send`, но не `Sync`), `transcribe_bytes`/`classify_bytes` передают сеймы в
> CPU-пайплайн; отказ `--npu-ops` в `serve` заменён семейными списками; бюджет
> контекстов проверяется и для ViT.
> E — GTE (`gte_new_rope_geglu`) использует те же пять кодов, что и BERT:
> `open_array_gte` больше не мёртвый код, оба отказа (`refuse_gte_array_ops`
> в семейном гейте и повторный в `embed::run`) удалены. `pipelines/gte.rs`
> перестал игнорировать eltwise: embedding-LN и обе post-LN идут через
> `eltwise.layernorm`, GEGLU-gate — через `eltwise.gelu`, внимание — через
> `array.attention` (softmax внутри блока, второй раз не диспатчится). Score-строки
> при Array-softmax пишутся с шагом `sc_cols`, хвост заполняется `-1e30`;
> проверки ширины — `SoftmaxTooNarrow` / `SoftmaxWidthMismatch`. `fix_width`
> и бюджет контекстов включены и для GTE. Mock-тест подтверждает 4 GEMM +
> attention + 3 LN + 1 GELU, host-путь побайтно прежний. Stage G закрыт,
Stage F частично закрыт (fft/mproj и logit — см. выше); остаются conv1/conv2,
`cross_kv`, depthwise conv инекратные head-проекции.

### Этап A. Общая инфраструктура открытия дизайнов и спеков

1. Вынести в общий модуль (например `cli/src/npu.rs`, либо расширить `cli/src/arrayconv.rs`) функции по образцу embedding-пути:
   - `open_array_text(container, path, geometry, root, args) -> Box<dyn ArrayText>` — копия `cli/src/embed.rs:983-1062`: проверка BF16-панелей, `crate::transport::open()` (`transport.rs:11-16`), сначала `ArrayDesign::open_from_container(path, set, transport)` (`xdna/array.rs:160`), потом `ArrayDesign::open(find_artifacts(...)...)` (`:123`), сборка `Vec<GemmSpec>` (`array_gemms.rs:121`) и `ArrayGemms::open` (`:258`).
   - `open_eltwise(...)` — по образцу `embed.rs:875-918` / `open_design` `:926-952`, один hw_context на каждый проход (`layernorm`, `gelu`, `softmax`), `Eltwise::with_rmsnorm` (`eltwise.rs:448`).
   - `open_convs(...)` для conv-семейств — уже есть `arrayconv.rs:34-101`; убрать хардкод `"gemm_rtp"` (`:66-97`) — сет должен приходить из контейнера/параметра.
2. Спеки для каждой модели строить из имён весов, как в `embed.rs:1043-1058` (qkv/attn_out/ffn_up/ffn_down на слой) и `pose::array_specs` (`pipelines/pose.rs:585`).
3. Проверки при открытии: `Datapath::of` (`gemm.rs:136-167`) допускает только `bf16/bf16` и `i8/i32`; для int8 нужны `<weight>.wscale` + `.asmooth` (`array_gemms.rs:346-368`); несовпадение `b_layout_hash` — опасный случай (совпадают размеры, но не layout) — сверять как в `core/npum-format/src/layout.rs:191-203` и `tools/lib/npue.py:1236-1240`.
4. `--artifacts` сейчас игнорируется в `embed::find_artifacts` (`embed.rs:562-584`); поправить единый поиск, иначе сообщения «Pass --artifacts …» (`embed.rs:945-947`) останутся ложными.

### Этап B. Сеймы в CPU-пайплайнах Whisper и ViT

Принцип: сигнатуры получают `Option<&mut dyn ArrayText>` / `Option<&mut dyn ArrayEltwise>`; при `None` путь остаётся **побайтно прежним**. Все ветки — по образцу `bert.rs:872-1412`.

1. `backends/cpu/src/pipelines/whisper.rs`:
   - `Encoder::run_into` (`:246`), `attention_into` (`:1196`), `Decoder::step` (`:632`), `greedy` (`:909`), `logits_from_into` (`:973`), `layernorm_rows` (`:1054`), `front_end` (`:1367`).
   - Ветки: QKV/attn_out/FFN-up/FFN-down → `array.gemm(...)`; attention → `array.attention(..., AttentionShape{...}, mask, softmax, ctx)`; LN/GELU/softmax → `eltwise.*`.
   - Оговорка: хостовый GELU Whisper — точный erf; дизайн `gelu/` — полином. Нужна отдельная проверка близости (см. `NPU_OPS.ru.md`, ячейка `stt/gelu`).
   - Протянуть параметры через `whisper_session.rs::transcribe_samples` (`:139`) и `run_window`.
2. `backends/cpu/src/pipelines/vit.rs`: `Vit::encode_into` (`:318`), `Vit::classify_into` (`:486`) — `ArrayTextRef` + `ArrayEltwise`; patch embed как GEMM (`:352-361`), 4 GEMM на слой, внимание, LN/GELU/softmax. Голову классификатора (1000 выходов) на массив **не переносить**: причина в `ops.rs:120-127`; код `logit` остаётся в отказе.
3. Защиты ширины score-строки и dispatch'ов — как в `bert.rs:89-102`, `:986-994`, `:1034-1044` (`SoftmaxTooNarrow`, `SoftmaxWidthMismatch`); тайминги и счётчики `dispatches`/`attn_dispatches` делать явными.

### Этап C. CLI-глаголы

1. `cli/src/stt.rs`: в `command` (`:592`) и `run` открывать два дизайна (`gemm_rtp` + `gemm_rtp_dec`, бюджетcontext'ов уже проверяется в `check_context_budget` `:347-357`) и eltwise-сеты; передавать сеймы в пайплайн. Заменить `refuse_array_not_ported(..., &[])` (`:608`) на реальный список: `gemm`, `attn`, `layn`, `gelu`, `softm`. Коды `conv/mproj/fft/logit` оставить в отказе до появления сеймов/дизайнов (см. этап F) — либо уточнить `Family::Stt` в `core/npum-families/src/ops.rs:43-57`.
2. `cli/src/vit.rs`: `command` (`:241`), `find_designs`, `print_block`; заменить `&[]` (`:271`) на `gemm,attn,layn,gelu,softm,conv`. `vit_refusal` (`ops.rs:104-130`) оставить для мёртвых кодов `mproj/fft/logit` и головы классификатора.
3. Статус/блок: сделать `places`-функции по образцу `embed::places` (`embed.rs:630-699`), `status::announce` вызывать до отказов; `staged_bytes` (`embed.rs:721-763`); `fix_width` (`embed.rs:373`) — дизайн фиксирует ширину score, поэтому батч не должен её сужать.
4. Conv-семейства (`pose/hands/mppose/ocr`) уже имеют `conv`; добавить только недостающие коды, если они появятся.

### Этап D. Сервер (`serve` + движки)

1. `server/src/engines.rs`: у каждого движка (`Transcriber :85`, `Classifier :268`, `PoseDetector :367`, `Embedder :496`) добавить:
   - опциональные поля с хендлами массива,
   - `open_with_xdna(...)`,
   - метод-инжектор типа `with_xdna_device(...)`.
   Образец — Vulkan-сейм `Embedder::open_with_vulkan` (`:560`) и `with_vulkan_device` (`:626`); в `embed_impl` (`:660`) ветка по маске кодов и передача сейма в CPU-пайплайн.
   Важно: хендлы транспорта `Send`, но **не `Sync`** (`xdna/transport.rs:24-29`) — хранить за `Mutex` (как `vulkan: Option<Arc<Mutex<Option<VulkanOps>>>>` в `engines.rs:511`), по одному на движок/слот пула декодеров.
2. Обработчики `server/src/routes.rs` (`transcriptions`, `classify`, `pose`) должны передавать сейм в запрос; `embed` уже передаёт `ops` в `embed_into`.
3. `cli/src/serve.rs`:
   - сделать XDNA-пробы по образцу Vulkan-пробы (`:572-581`): открыть дизайны один раз при старте, проверить `has_gpu`-аналог (наличие транспорта/дизайна), инъекция в движки (`:308-314`, `:587`).
   - бюджет контекстов проверять для **всех** много-дизайнных семей (`require_context_budget`, `contention.rs:225`), а не только STT (сейчас `stt.rs:350`).
   - **Не удалять** отказ `--npu-ops` (`serve.rs:107-115`) до тех пор, пока сейм семейства не реализован; затем заменить `&[]` на список реально поддерживаемых кодов. Это ужесточение было внесено в рамках багфикса и является предохранителем от «командная строка говорит array, а запуск считает на хосте».

### Этап E. GTE

1. Включить `open_array_gte` (`embed.rs:1064-1138`) и убрать отказ `refuse_gte_array_ops` (`families.rs:178-192`) только после того, как сейм заработает.
2. В `pipelines/gte.rs`: перестать игнорировать `_eltwise` (`:557`), добавить `array.attention` (сейчас только `fused_attention`/host-softmax, `:690-722`), ширинные проверки, поддержку MQA и banded attention (Gemma: 3 головы запроса на 1 KV, sliding window 512 при упаковке на 2048).

### Этап F. Операции, которых на массиве нет

Сейчас в `backends/xdna/src/dispatch.rs:49-135` — заглушки для `conv1d`, `conv2d`, `mproj`, `fft`, `logit`; отдельного array-опера для mel-банка нет (он в `frontend/whisper_features.rs:457`). Для «полностью на NPU» нужны:

- новые дизайны/потоки для `conv1/conv2` Whisper, mel-банка, 400-точечного FFT, связанного эмбеддинга-логитов;
- новые трейты/методы в `core/npum-model/src/array.rs` и их реализации в `backends/xdna`;
- для depthwise/transposed conv — отдельный путь: `ArrayConvs` явно их не поддерживает (`array.rs:317-324`), `array_convs.rs` не проверяет `Datapath::of`.

Пока этого нет — оставить операции на хосте и **называть их в статусе**, не принимать `--npu-ops` молча. Приоритет по выгоде: `gemm` (уже даёт 0.241 с против 0.435 с на ViT) > `conv` > eltwise/attn (замеры в `NPU_OPS.ru.md` показывают, что attn на массиве медленнее хоста).

### Этап G. Тесты и гейты

1. Новые тесты паритета (device-gated, с skip как в `backends/xdna/tests/*`):
   - Whisper: энкодер/декодер через сеймы — mel-величины, hidden, токены greedy; отдельная оговорка для `gelu_erf` против дизайн-GELU.
   - ViT: patch-embed GEMM, hidden по слоям, logits/top-1, счётчик диспатчей (ожидаемо 48 GEMM + 1 patch embed).
   - Server: те же проверки через `Transcriber`/`Classifier`/`PoseDetector`, а не только через CLI; для pose — подтвердить, что depthwise остался на хосте и посчитан.
   - int8: `.asmooth`/`.wscale` end-to-end, int8 attention → host-fallback.
   - `require_context_budget`: `tool_ran=false` и `--allow-contention`; отказ по generation/dtype через `DesignSet::fits`.
2. Пороги: `BF16_HALF_ULP = 1/512`, `MAX_RELATIVE = 4*half_ulp`, `MAX_ONE_MINUS_COS = 2e-3` (`tests/design_numerics.rs:34-42`). Точного совпадения с CPU нет (`caps.bit_exact_with_cpu=false`, `lib.rs:92-95`).
3. CLI-гейты: `tools/verify/verify_cli_flags.py`, `verify_endpoint.py`, `verify_npu_op_matrix.py`, `verify_pose_streamset.py`; проверить, что статус-блок называет array только реально диспатчимые проходы.

## 3. Критерии приёмки

- `npuimage classify … --npu-ops gemm` реально диспатчит GEMM и в статусе стоит `on the ARRAY`; паритет в пределах порогов; `classify` через `serve` делает то же.
- `npuaudio transcribe … --npu-ops gemm,layn,gelu,softm,attn` и `npuaudio serve …` с теми же кодами идут через сеймы; два контекста (`gemm_rtp` + `gemm_rtp_dec`) проверены бюджетом.
- Непортированные коды (`mproj`, `fft`, `logit`, depthwise conv, GTE до этапа E) по-прежнему отвергаются **по имени**, а не игнорируются.
- Все существующие тесты (`cargo test --workspace`) и модельные гейты (`verify_tail.py`) остаются зелёными.

## 4. Риски

- `Geometry::context_budget` — `None` для обоих поколений (`geometry.rs:84`): «бюджет неизвестен» не означает «место есть». Сейчас лимит — env `NPU_HWCTX_LIMIT` или 6 для npu1 (`contention.rs:93-119`). Eltwise — по одному контексту на проход, поэтому Whisper/ViT с 4 дизайнами легко превышают заложенные 2 контекста.
- Дизайн-сеты individual per model и только под npu1; npu2-наборов и платы в дереве нет (`tests/generations.rs:15-19`). Без корректного сета открытие обязано возвращать ошибку, а не считаться на хосте.
- int8 attention отвергнут в `attention.rs:268-275`; conv-панели — bf16; `array_convs.rs` не проверяет `Datapath::of`.
- Таймаут диспатча требует переоткрытия контекста (`direct.rs:1260-1265`, `NPU_DISPATCH_TIMEOUT_MS`).
- Известные неполные места после A–E (не «тихие» отказы по коду, а отсутствующие потоки в наличных дизайнах): `cross_kv` декодера Whisper считается на хосте при `set_source` — в набор `gemm_rtp_dec` нет потока для этого кэша, и его нужно добавить в этапе F; хостовый GELU Whisper — точный `erf`, а дизайн `gelu/` — полином, поэтому численное сближение для `stt/gelu` проверяется отдельно в этапе G.
- **Gemma: токенизатор.** `tokenizer.gemma_table` разбирается как таблица кусков GEMATOK1, но Rust-токенизатор всё ещё побайтовый/посимвольный, а не SentencePiece BPE. Текстовые запросы через `embed`/`serve` дают корректный.forward для полученных id, но id отличаются от HF; паритет с эталоном проверялся на фиксированных id, а не на реальном тексте.
- **Gemma: MQA attention** (QK^T и softmax·V) по-прежнему на хосте: `ArrayAttention` не умеет Q/K-нормы и RoPE перед оценкой. Проекции и softmax идут на массив; коды `layn/gelu/attn` честно отвергаются. Маска sliding window точна только до 512 токенов (дальше модель отвергается), как и оговорено в контейнере.

---

Примечание: все ссылки на строки даны по состоянию рабочего дерева на момент написания плана (после багфиксов); при реализации сверять с актуальным кодом.
