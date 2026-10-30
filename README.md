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
| LoRA-сложение + RoPE + раскладка | [`kernels/rope.py`](solution/kernels/rope.py) | `_rope_lora_scatter_fwd_kernel`: `fp32(base) + fp32(lora)`, поворот `q·cos + rotate_half(q)·sin` в fp32 и запись **сразу в fp16** в `[B, H, S, D]`, который читает SDPA: в базе autocast округляет fp32-результат RoPE до fp16 ровно в этой точке, поэтому значения те же, а fp32-копии q/k/v и трёх кернелов приведения нет (в обе стороны). Backward (`_rope_gather_bwd_kernel`) читает fp16-градиент SDPA, применяет сопряжённый поворот (`-sin`) в fp32 — активации RoPE не хранятся. GQA: одна компиляция на q и k/v. |
| SwiGLU в fp16 | [`kernels/swiglu.py`](solution/kernels/swiglu.py) | `silu(g)·u` из fp16 выходов GEMM, математика в fp32, сохраняются только `g` и `u`, backward пишет `dg`, `du` поверх них. *Это вариант SwiGLU из примера стартового кита — в долю своих кернелов мы его не засчитываем.* |
| NF4 + double quant | [`kernels/nf4.py`](solution/kernels/nf4.py) | `_nf4_dequant_nested_kernel` за один проход делает то, на что bitsandbytes тратит три кернела: декодирует 8-битный absmax (`code2[a8]·absmax2 + offset`), оба полубайта через таблицу NF4 и пишет fp16-веса. Результат **побитно** равен `bitsandbytes.functional.dequantize_4bit` (FMA выключен, как у двух отдельных кернелов bnb). Запускается 392 раза за шаг (7 проекций × 28 слоёв × forward/backward). |
| Удаление pad-токенов | [`patches/patch_qwen.py`](solution/patches/patch_qwen.py) | Все потокенные операции (нормы, проекции, LoRA, MLP, lm_head, loss) видят только реальные токены. Attention считается тем же вызовом SDPA из transformers на паддированной раскладке с той же маской, что и в базе. Вход LoRA-dropout на паддированной раскладке строится один раз на слой и общий для q и v (маски dropout от этого не меняются: они зависят только от формы и порядка вызовов). |
| Пересчёт gate/up в backward (по умолчанию) | [`kernels/nf4.py`](solution/kernels/nf4.py) | Вместо двух fp16-тензоров `[токены, 8960]` (~35 КБ на токен на слой, ~1,7 ГБ активаций на самом длинном батче) MLP сохраняет только fp16-вход и в backward заново делает два GEMM. Веса gate/up распаковываются в backward один раз и служат и пересчёту, и градиенту по входу, так что пересчёт стоит два GEMM и ни одной лишней распаковки NF4. Тот же вызов cuBLAS на тех же входах даёт те же gate/up. Выключается `SOUP_KERNELS_MLP_RECOMPUTE=0`. |
| Хранение таблицы эмбеддинга в fp16 | [`patches/patch_qwen.py`](solution/patches/patch_qwen.py) | Связанные `embed_tokens`/`lm_head` хранятся один раз в fp16 (467 МБ вместо 933 МБ) — только если `fp16 → fp32` возвращает fp32-таблицу бит в бит (так и есть: Soup грузит модель в fp16, peft лишь повышает точность). Строки эмбеддинга по-прежнему выдаются в fp32; после обучения fp32-таблица возвращается. |
| Прекомпиляция | [`warmup.py`](solution/warmup.py) | Кернелы компилируются в потоке, пока Soup грузит модель (кэш Triton в каждом замере пустой, а компиляция входит во время). Поток не трогает RNG torch (от него зависят маски dropout и порядок данных). |

### Как сохраняется качество

Мы не меняем ни одной точки округления базового Soup: остаточный поток в fp32, входы GEMM в fp16,
сложение LoRA в fp32, входы SDPA в fp16, `dL/dz` округляется до fp16 в том же масштабе, что и в
базе (градиент берётся в forward, поэтому тренер передаёт текущий масштаб `GradScaler`, а backward
умножает на `grad_output / scale`, что в обучении ровно 1,0). Маски LoRA-dropout тянутся на
паддированной раскладке `[B·S, H]`, поэтому случайный поток совпадает с базой бит в бит. Отличаются
только порядок части fp32-сумм и M у GEMM (меньше строк) — это уровень округления, а не математики.

Как это проверено — в разделе [«Проверка»](#проверка) ниже (с указанием, что запускалось
на CPU, а что ещё предстоит замерить на T4).

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
├── test_kernels.py        на GPU: кернелы против PyTorch / bitsandbytes и сквозной тест
└── test_interpret.py      на CPU (TRITON_INTERPRET=1): те же @triton.jit-кернелы и сквозной тест
```

Переключатели (переменные окружения, по умолчанию всё включено): `SOUP_KERNELS=0` — чистый Soup;
`SOUP_KERNELS_UNPAD=0`, `SOUP_KERNELS_FP16_EMBED=0`, `SOUP_KERNELS_WARMUP=0`,
`SOUP_KERNELS_CE_CHUNK=256`, `SOUP_KERNELS_MLP_RECOMPUTE=0` (хранить gate/up вместо пересчёта:
быстрее, но больше памяти), `SOUP_KERNELS_VERBOSE=1` (печатать причины отката на обычный путь).

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
python -m pytest solution -q                   # кернелы против эталона (на GPU)
TRITON_INTERPRET=1 python -m pytest solution/test_interpret.py -q   # то же на CPU, без GPU
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

## Проверка

Всё ниже реально запускалось (Windows, CPU, Python 3.12, torch 2.15 CPU, Triton 3.6.0
(`triton-windows`), soup-cli 0.77.0, transformers 5.16.1, peft 0.20.0, trl 0.29.1,
bitsandbytes 0.50.2, accelerate 1.14.0). На T4 пока **не** запускалось — см. «Результаты».

| Что | Команда | Итог |
|---|---|---|
| Сами `@triton.jit`-кернелы в интерпретаторе Triton на размерах модели (hidden 1536, 12/2 головы × 128, словарь 151 936, MLP 8960), forward + backward против PyTorch; NF4 — побитно; argmax с равными максимумами через границы тайлов; сквозной тест крошечной 4-битной Qwen2 + LoRA (dropout 0,05, батч с pad) против базовых transformers/peft/bitsandbytes | `TRITON_INTERPRET=1 python -m pytest solution/test_interpret.py` | **10 passed** |
| То же сквозное сравнение, числа | (сквозной тест выше) | loss: отн. разница 2·10⁻⁶; энтропия совпадает до 10⁻⁶; поток RNG после шага **идентичен** (те же маски LoRA-dropout); 8 градиентов LoRA — худшая отн. разница 1,1·10⁻³ |
| Самопроверка `warmup.compile_kernels` (то, что на T4 включает/выключает быстрый путь) на реальных размерах Qwen2.5-1.5B | интерпретатор | `problems: []` |
| Старый CPU-набор `reports/test_cpu_validation.py` | с `TRITON_INTERPRET=1` / без (PyTorch-откат) | **53 passed** / **53 passed** |
| 79 регрессионных тестов Soup с `solution.apply()` (проверка 10) | `SOUP_APPLY_SOLUTION=1 python -m pytest tests` | **79 passed** |
| Статическая часть `check_rules.py`: свои кернелы, запрещённые импорты | `check_rules.source_kernels` | 8 своих кернелов, запрещённых импортов нет; `_swiglu_fp16_fwd_kernel` похож на пример на 0,82 (жюри читает руками — мы его и так не засчитываем) |

Важно: до этой ревизии кернелы **ни разу не исполнялись** — на машине прошлой проверки не было ни
GPU, ни Triton, все тесты `solution/test_kernels.py` были пропущены, а «CPU-проверка» шла по
PyTorch-откату. Интерпретатор исполняет именно код кернелов (маски, индексы, редукции, запись на
месте), но не компилирует их под sm_75: компиляцию и скорость проверяет только запуск на T4
(`python -m pytest solution` в ноутбуке, затем `harness/run_task.py`).

Чтобы повторить на CPU: тот же набор пакетов, `numpy<2.3` (интерпретатор Triton 3.6 не работает с
NumPy ≥ 2.5) и `TRITON_INTERPRET=1`.

## Результаты

**На T4 ещё не замерено — заполнить из `runs/result.json` и `runs/checks.json` сохранённой версии
Kaggle-ноутбука.** Никаких чисел скорости и памяти ниже нет, пока их не даст настоящий запуск.

| | База Soup 0.77.0 | Решение | Отношение |
|---|---|---|---|
| Время `soup train`, с | НЕ ЗАМЕРЕНО | НЕ ЗАМЕРЕНО | speedup ×… |
| Пик памяти GPU (nvidia-smi), МиБ | НЕ ЗАМЕРЕНО | НЕ ЗАМЕРЕНО | база/решение ×… |
| Eval loss (публичные строки) | НЕ ЗАМЕРЕНО | НЕ ЗАМЕРЕНО | … % к базе |
| Доля наших кернелов (проверка 4) | — | НЕ ЗАМЕРЕНО | (без SwiGLU: …%) |

Оценка по разбору кода, **не результат** (нужна, чтобы знать, что проверять на T4):

- память: сохранённые активации ≈ 21 КБ на реальный токен на слой (fp32-остаток двух норм 12 КБ,
  входы LoRA 6 КБ, fp16-вход MLP 3 КБ) + ≈ 15 КБ на паддированный токен на слой в attention; на
  самом длинном батче (~1900 реальных токенов из 2048) это ~2 ГБ на 28 слоёв, плюс ~1,2 ГБ модели
  (fp16-таблица эмбеддинга + 4-битные слои) и контекст CUDA — пик порядка 4 ГБ против ~14,3 ГБ у
  базы. Без пересчёта MLP (`SOUP_KERNELS_MLP_RECOMPUTE=0`) — примерно на 1,7 ГБ больше;
- время шага: GEMM на реальных токенах вместо паддированных, нет fp32-цепочек и логитов
  `[B, S, V]`; пересчёт MLP добавляет 2 GEMM на слой в backward (оценка +15–20% к шагу). Время
  всей команды сокращается меньше, потому что загрузка модели не меняется;
- доля наших кернелов без SwiGLU: распаковка NF4 ~26 мс/шаг + CE ~4 мс + нормы/RoPE ~3 мс — по
  оценке 7–9% времени шагов 3+. Если на T4 выйдет ближе к 5%, это первое, что надо проверить.

Что сравнить на Kaggle перед финальной версией: `python benchmarks/verify_t4_speedup.py` с
`SOUP_KERNELS_MLP_RECOMPUTE=1` (по умолчанию) и `=0` — по формуле баллов
(35·ln s / ln s_best + 30·ln m / ln m_best) пересчёт выгоден, если он снижает пик памяти заметно
сильнее, чем замедляет запуск.

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
- Пересчёт MLP в backward включён по умолчанию по оценке (−~1,7 ГБ пика ценой двух GEMM на слой);
  на T4 это ещё нужно подтвердить A/B-замером (см. «Результаты»).
- Слой 0: в базе его q/k/v видят вход без градиента, и bitsandbytes считает их тем же
  dequant + `F.linear` при M > 64 (на T4), как и мы, но на паддированном M; у нас M — число
  реальных токенов, поэтому cuBLAS может выбрать другой алгоритм: отличие на уровне округления fp16.
- Если на машине жюри какой-то кернел не скомпилируется или разойдётся с эталоном PyTorch
  (самопроверка в `warmup.py` при старте обучения), быстрый путь выключается и запуск идёт как
  обычный Soup — без падения, но и без ускорения (тогда не пройдут проверки 3–4).
- Тесты кернелов лежат в `solution/test_kernels.py`, а не в `tests/`: `check_rules.py` требует
  ровно 79 тестов Soup в `tests/`, а папку `tests/` менять нельзя.
