# Soup Kernel Challenge — Triton-кернелы для `soup train` на Kaggle T4

Решение для кейса [Soup](https://github.com/MakazhanAlpamys/Soup) на Decentrathon 6.0: тот же
запуск `soup train` (Qwen2.5-1.5B-Instruct, NF4, LoRA r=16 на `q_proj`/`v_proj`, batch 4, до 512
токенов, 64 шага) на бесплатной T4 — быстрее и с меньшим пиком памяти, с тем же адаптером.
Всё решение — пакет [`solution/`](solution/), подключается одной функцией `solution.apply()`.

- Kaggle-ноутбук (сохранённая версия с результатами): **TODO — ссылка**
- Финальный коммит: **TODO — ссылка**

## Что сделано и почему

Отправная точка — профиль базового Soup из условия: ~36% времени шага — матричные умножения,
~44% — мелкие поэлементные операции и приведения типов, ~7% — softmax/CE по словарю из 151 936
токенов, ~5% — attention, ~5% — распаковка NF4; пик памяти ~14,3 из 15 ГБ. Разбор исходников
Soup 0.77.0 / transformers 5.16.1 / peft 0.20.0 / bitsandbytes 0.50.2 показал, откуда это берётся:

1. **Логиты `[B, S, V]` живут в нескольких копиях.** `ForCausalLMLoss` делает fp32-копию и
   сохраняет `log_softmax`, accelerate конвертирует выход модели в fp32, TRL для точности делает
   ещё одну `contiguous`-копию, backward выделяет два fp32-градиента. При 2048 токенах каждая
   копия — 1,24 ГБ.
2. **Остаточный поток и MLP в fp32.** `prepare_model_for_kbit_training` держит эмбеддинг и нормы
   в fp32, а bitsandbytes возвращает проекцию в dtype входа, поэтому RMSNorm, RoPE, сложение LoRA и
   SwiGLU идут цепочками fp32-кернелов PyTorch, а MLP сохраняет для backward три fp32-тензора
   `[токены, 8960]` на слой.
3. **Паддинг.** Батч дополняется до самой длинной строки (в среднем ~220 токенов на строку, до 512):
   заметная доля всей работы тратится на pad-токены.

| Что | Где | Как устроено |
|---|---|---|
| lm_head + CE + метрики по чанкам | [`kernels/cross_entropy.py`](solution/kernels/cross_entropy.py) | Логиты считаются кусками по 256 токенов (cuBLAS fp16). Кернел `_cross_entropy_fwd_bwd_inplace_kernel` (одна программа на токен) за один проход по словарю считает online max / sum-exp в fp32, энтропию (`entropy` TRL), argmax (`mean_token_accuracy` TRL) и CE токена (потери строк для `rewind.jsonl`), вторым проходом **на месте** записывает `dL/dz` поверх логитов, затем cuBLAS даёт градиент по скрытому состоянию. Тензор `[B, S, V]` не создаётся вообще. Градиент строится только для строк с меткой (они сортируются вперёд). |
| RMSNorm + сложение остатка | [`kernels/rmsnorm.py`](solution/kernels/rmsnorm.py) | `_add_rmsnorm_fwd/bwd_kernel`: `resid + delta` (fp32 + fp16) и нормализация за один проход, `rstd` в fp32; backward по замкнутой формуле, сразу отдаёт градиент и остатку, и fp16-выходу блока. |
| LoRA-сложение + RoPE + раскладка | [`kernels/rope.py`](solution/kernels/rope.py) | `_rope_lora_scatter_fwd_kernel`: `fp32(base) + fp32(lora)`, поворот `q·cos + rotate_half(q)·sin` и запись сразу в `[B, H, S, D]`, который ждёт SDPA. Backward (`_rope_gather_bwd_kernel`) применяет сопряжённый поворот (`-sin`) к градиенту — активации RoPE не хранятся. GQA: одна компиляция на q и k/v. |
| SwiGLU в fp16 | [`kernels/swiglu.py`](solution/kernels/swiglu.py) | `silu(g)·u` из fp16 выходов GEMM, математика в fp32, сохраняются только `g` и `u`, backward пишет `dg`, `du` поверх них. *Это вариант SwiGLU из примера стартового кита — в долю своих кернелов мы его не засчитываем.* |
| NF4 + double quant | [`kernels/nf4.py`](solution/kernels/nf4.py) | `_nf4_dequant_nested_kernel` за один проход делает то, на что bitsandbytes тратит три кернела: декодирует 8-битный absmax (`code2[a8]·absmax2 + offset`), оба полубайта через таблицу NF4 и пишет fp16-веса. Результат **побитно** равен `bitsandbytes.functional.dequantize_4bit` (FMA выключен, как у двух отдельных кернелов bnb). Запускается 392 раза за шаг (7 проекций × 28 слоёв × forward/backward). |
| Удаление pad-токенов | [`patches/patch_qwen.py`](solution/patches/patch_qwen.py) | Все потокенные операции (нормы, проекции, LoRA, MLP, lm_head, loss) видят только реальные токены. Attention считается тем же вызовом SDPA из transformers на паддированной раскладке с той же маской, что и в базе. |
| Хранение таблицы эмбеддинга в fp16 | [`patches/patch_qwen.py`](solution/patches/patch_qwen.py) | Связанные `embed_tokens`/`lm_head` хранятся один раз в fp16 (467 МБ вместо 933 МБ) — только если `fp16 → fp32` возвращает fp32-таблицу бит в бит (так и есть: Soup грузит модель в fp16, peft лишь повышает точность). Строки эмбеддинга по-прежнему выдаются в fp32; после обучения fp32-таблица возвращается. |
| Прекомпиляция | [`warmup.py`](solution/warmup.py) | Кернелы компилируются в потоке, пока Soup грузит модель (кэш Triton в каждом замере пустой, а компиляция входит во время). Поток не трогает RNG torch (от него зависят маски dropout и порядок данных). |

### Как сохраняется качество

Мы не меняем ни одной точки округления базового Soup: остаточный поток в fp32, входы GEMM в fp16,
сложение LoRA в fp32, входы SDPA в fp16, `dL/dz` округляется до fp16 в том же масштабе, что и в
базе (градиент берётся в forward, поэтому тренер передаёт текущий масштаб `GradScaler`, а backward
умножает на `grad_output / scale`, что в обучении ровно 1,0). Маски LoRA-dropout тянутся на
паддированной раскладке `[B·S, H]`, поэтому случайный поток совпадает с базой бит в бит. Отличаются
только порядок части fp32-сумм и M у GEMM (меньше строк) — это уровень округления, а не математики.

Проверено локально (CPU, интерпретатор Triton, те же версии transformers/peft/trl/bitsandbytes,
крошечная 4-битная Qwen2 + LoRA): loss совпадает до 2·10⁻⁷, энтропия и потери строк — точно,
**все градиенты LoRA совпадают бит в бит** (с dropout и удалением pad); через настоящие TRL
`SFTTrainer` + `RewindSFTTrainer` из Soup совпадают имена метрик, `num_tokens`, записи
`rewind.jsonl`; все 79 регрессионных тестов Soup проходят с `SOUP_APPLY_SOLUTION=1`.

### Как подключается

`solution.apply()` ставит патчи на уровне классов (ищутся в момент вызова):

- `Qwen2ForCausalLM.forward` — быстрый путь берётся только в обучающем шаге, запрошенном тренером,
  и только для конфигурации этой задачи (NF4 + LoRA на q/k/v, SDPA, fp16 autocast). Всё остальное
  (eval, генерация, другие модели) уходит в исходный forward;
- `trl.SFTTrainer.compute_loss` (его наследует `RewindSFTTrainer` из Soup) — логирует `entropy`,
  `num_tokens`, `mean_token_accuracy` по формулам TRL из значений кернела и кладёт потери токенов
  туда, где их читает rewind Soup для шага без логитов (`_soup_chunked_step`, тот же механизм, что
  у встроенного в Soup chunked loss #1693);
- `SFTTrainerWrapper.setup` / `.train` из Soup — запуск прекомпиляции и подготовка модели.

Структура пакета:

```text
solution/
├── __init__.py            экспортирует apply() (харнесс вызывает solution.apply())
├── apply.py               apply(): единственная точка входа, ставит патчи
├── kernels/               @triton.jit-кернелы и их autograd-функции
│   ├── rmsnorm.py         residual add + RMSNorm (fwd + bwd)
│   ├── rope.py            LoRA add + RoPE + раскладка под SDPA (fwd + bwd)
│   ├── swiglu.py          SwiGLU в fp16, backward на месте (не засчитываем как свой)
│   ├── cross_entropy.py   lm_head + CE + телеметрия TRL / rewind кусками
│   └── nf4.py             распаковка NF4 + 4-битные gate/up/down
├── patches/patch_qwen.py  быстрый forward Qwen2 и хуки тренера / Soup
├── warmup.py              компиляция и самопроверка кернелов во время загрузки
├── _runtime.py            переключатели SOUP_KERNELS_* и проверки GPU / Triton
├── swiglu.py              пример из стартового кита, как есть (apply() его не вызывает)
└── test_kernels.py        кернелы против PyTorch / bitsandbytes и сквозной тест
```

Переключатели (переменные окружения, по умолчанию всё включено): `SOUP_KERNELS=0` — чистый Soup;
`SOUP_KERNELS_UNPAD=0`, `SOUP_KERNELS_FP16_EMBED=0`, `SOUP_KERNELS_WARMUP=0`,
`SOUP_KERNELS_CE_CHUNK=256`, `SOUP_KERNELS_MLP_RECOMPUTE=1` (пересчёт gate/up в backward: меньше
памяти, медленнее), `SOUP_KERNELS_VERBOSE=1` (печатать причины отката на обычный путь).

### Источники идей

- Online softmax: M. Milakov, N. Gimelshein, *Online normalizer calculation for softmax*, 2018.
- Логиты кусками без материализации: Cut Cross-Entropy (E. Wijmans et al., 2024) и собственный
  `stream_loss_chunk` Soup (#1693) — идея, код наш.
- NF4 и double quantization: QLoRA (T. Dettmers et al., 2023), формат bitsandbytes.
- Сопряжённый поворот в backward RoPE, слияние residual + RMSNorm — общеизвестные приёмы
  (Liger-Kernel, Unsloth делают похожее; их код не использовался и не импортируется).

## Как запустить

На Kaggle (так проверяет жюри): импортировать `templates/kaggle_notebook.ipynb`, Accelerator
**GPU T4 x2**, Internet on, вписать URL репозитория в первую ячейку, **Run All** (~10 минут).
Ноутбук прогоняет `pytest solution` (наши кернелы против PyTorch / bitsandbytes и сквозной тест на
маленькой 4-битной Qwen2), затем базу и решение, `check_rules.py` и `score.py`.

Локально на Linux с CUDA-GPU:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py                    # data/train.jsonl, data/eval_public.jsonl
python -m pytest solution -q                   # кернелы против эталона
python benchmarks/verify_t4_speedup.py         # быстрый A/B: 15 шагов база vs решение
python harness/run_task.py --repeats 2         # замер как у жюри -> runs/result.json
python harness/check_rules.py                  # 10 проверок -> runs/checks.json
python harness/score.py runs/result.json
```

`benchmarks/verify_t4_speedup.py` запускает каждую сторону в отдельном процессе с окружением
харнесса, останавливает обучение после N шагов (`--steps`, по умолчанию 15) и печатает токены/с
(шаги 3+), пик `torch.cuda.max_memory_allocated` / `max_memory_reserved` и пик nvidia-smi, а
кривые loss / entropy / accuracy сверяет через `torch.testing.assert_close` и по критериям
`check_rules.py` (первый шаг ≤ 1%, среднее ≤ 3%).

## Результаты

**TODO: заполнить из `runs/result.json` и `runs/checks.json` сохранённой версии Kaggle-ноутбука.**

| | База Soup 0.77.0 | Решение | Отношение |
|---|---|---|---|
| Время `soup train`, с | | | speedup ×… |
| Пик памяти GPU (nvidia-smi), МиБ | | | база/решение ×… |
| Eval loss (публичные строки) | | | … % к базе |
| Доля наших кернелов (проверка 4) | — | …% | (без SwiGLU: …%) |

Оценка по разбору памяти и трафика, **до замера на T4** (ориентир, не результат): пик ~5 ГБ против
~14,3 ГБ у базы; шаг обучения примерно вдвое быстрее за счёт удаления pad-токенов и слияния
fp32-цепочек; общее время команды сокращается меньше, потому что загрузка модели (~20 с) не
меняется. Доля наших кернелов без SwiGLU — по оценке ~10% времени шагов 3+.

## Ограничения и что дальше

- Быстрый путь покрывает именно эту конфигурацию: Qwen2, NF4 с double quant, LoRA на q/k/v без
  DoRA и других вариантов, SDPA, fp16 autocast. Иначе — обычный путь Soup (с теми же метриками).
- Attention остаётся SDPA из PyTorch на паддированной раскладке (на T4 — memory-efficient ядро);
  собственный varlen FlashAttention для sm_75 с пропуском pad-тайлов — следующий шаг.
- Сравнение побитной точности делалось на CPU в интерпретаторе Triton; на T4 порядок суммирования
  в cuBLAS при другом M даёт отличия на уровне округления fp16 — это то, что проверяют
  `pytest solution` и `check_rules.py`.
- Слияние LoRA с распакованными весами (`W + s·B·A`) не делается намеренно: в задаче dropout 0,05
  стоит только на входе LoRA, поэтому такое слияние меняет функцию в режиме обучения.
- Fused NF4 GEMM (распаковка весов в регистрах прямо перед `tl.dot`, как в Marlin / TileLang)
  не делали: на T4 при M ≈ 900–2048 строк cuBLAS по fp16-весам быстрее Triton-GEMM на Turing,
  а плотная копия весов живёт только внутри одной проекции (до 27,5 МБ), так что по памяти
  выигрыша почти нет. Вместо этого три кернела bitsandbytes слиты в один (`kernels/nf4.py`).
- Layer streaming (`StreamedLargeLayer`, `_prefetch_stream`) в этой задаче не используется, а
  переключать встроенные режимы Soup правила запрещают — эти пути не трогали.
- Пересчёт MLP в backward (`SOUP_KERNELS_MLP_RECOMPUTE=1`) по оценке снижает пик ещё на ~1,5 ГБ
  ценой ~10% скорости шага; по умолчанию выключен — решать по замеру.
- Если на машине жюри какой-то кернел не скомпилируется или разойдётся с эталоном PyTorch
  (самопроверка в `warmup.py` при старте обучения), быстрый путь выключается и запуск идёт как
  обычный Soup — без падения, но и без ускорения (тогда не пройдут проверки 3–4).
- Тесты кернелов лежат в `solution/test_kernels.py`, а не в `tests/`: `check_rules.py` требует
  ровно 79 тестов Soup в `tests/`, а папку `tests/` менять нельзя.
