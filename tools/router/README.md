# Embedding task router (issues #28, #35) - research scripts

Goal: classify each message as `general` / `math` / `code` / `translate` in any language, in < 20 ms, with a model far
smaller than Laya (614 MB), then let the measured catalog scores pick the model.

| file | what |
|---|---|
| `check_embed.py` | llama.cpp GGUF embeddings vs PyTorch (cosine 0.9998-0.9999 for e5-small Q8_0, see below) |
| `build_data.py` | training data labelled by source (MGSM, GSM8K, MBPP, CRUXEval, FLORES, Aya, oasst2, Global-MMLU, ThaiExam) |
| `gen_data.py` | short everyday-style requests per class in 15 languages, written by a local LLM (`generated.jsonl`) |
| `train_router.py` | embed with llama-server `--embedding --pooling mean`, logistic-regression head, eval vs keyword rules, `--export` the head |
| `eval_package.py` | the router exactly as localllm runs it (`localllm/taskclf.py`): accuracy per order, latency, gateway reasons |
| `eval_laya.py` | Laya (multilingual checkpoint, zero-shot choice) on the same test set |
| `shared_test.jsonl` | 144 hand-written requests, 15 languages, never used for training |

Model: `intfloat/multilingual-e5-small` (MIT) converted with `convert_hf_to_gguf.py` after setting
`architectures: ["XLMRobertaModel"]` and `pad_token_id: null` in config.json (BERT body + XLM-R sentencepiece tokenizer;
a pad_token_id makes the converter drop the first position-embedding row, shifting positions by one). Deleting the key
is not enough with llama.cpp b11457: its converter reads the config through transformers, whose BertConfig default (0)
comes back - the log then says `context length = 511`. Measured on CI (cosine vs PyTorch, 6 texts): key deleted
0.9922-0.9996, null 0.9998-0.9999. Q8_0 = 126 MB. Inputs need the `query: ` prefix, cap ~450 chars.

**Current result** (training data + 2,833 short everyday requests in 15 languages written by gemma-4 with `gen_data.py`;
`train_router.py --with-generated`; head in `router_head.json`, 4 x 384 logistic regression on e5-small Q8_0):

| | shared test (144) | items least like training (nearest cosine < 0.90, n=73) | latency, CPU | size |
|---|---|---|---|---|
| **embedding router** | **97.9%** | **98.6%** | 7.3 ms median | 126 MB |
| Laya multilingual (zero-shot) | 77.8% | 71.2% | 103 ms median | 614 MB |
| keyword rules (`detect_task`) | 56.9% | - | < 1 ms | 0 |

Caveat: the shared test set is written by an AI (Claude) too, and generated training items are AI-written; 10 of 144 test
items have a training item with cosine > 0.95 (the filtered column removes such cases). A human-written test set from
native speakers is the next proof needed.

First results, before the generated data (RX 9070 XT PC, CPU, shared test): Laya 77.8% (103 ms median) | embedding head 64.6% | embedding head with
keyword rules first 72.2% (~6 ms) | keyword rules alone 56.9%. Held-out (same style as training) 98.6%: the gap is
training-data style, being fixed with generated short requests.

CPU data points (CI, GitHub ubuntu-latest 4 vCPU, llama.cpp b11457, `pad_token_id: null`), shared test through
`localllm/taskclf.py` exactly as shipped (own CPU llama-server, pure-Python head; per message = embed + head):

| head | embedding only | keyword rules first | keyword rules alone | per message, median / p95: EPYC 7763; Xeon 8370C |
|---|---|---|---|---|
| committed `router_head.json` | 97.9% | **98.6%** | 56.9% | 8.3 / 14.3 ms; 12.7 / 26.0 ms |
| trained in CI, with `generated.jsonl` | 97.2-97.9% | **97.9-98.6%** | 56.9% | 8.6 / 15.1 ms; 13.2 / 26.5 ms |
| trained in CI, without generated requests | 66.7% | 75.0% | 56.9% | 8.9 / 15.0 ms; - |

The head trained in CI moves by one message between the two runner CPUs (float differences in the embeddings, so a
slightly different fit); keyword rules first is ahead of embedding only in every run. The first message also starts the
embedding server: 1.3-1.5 s, once.

## In the package (#35)

`localllm serve --models auto` decides each message's task with `router.classify_task()`: the keyword rules first (they
only commit on a translate/math cue), the embedding classifier otherwise (`router.TASK_ORDER`, one line to change).
`localllm/taskclf.py` is standard library only: it starts its own llama-server on the CPU (`--embedding --pooling mean
-ngl 0 -dev none`, free private port) on the first message that needs it, reuses it, stops it when localllm exits, and
runs the head in pure Python. Files, in `~/.localllm/models/` (or a `$LOCALLLM_MODELS` folder):

- `multilingual-e5-small-Q8_0.gguf` and `router_head.json` (from `train_router.py --export`)
- downloaded on first use from `taskclf.GGUF_URL` / `taskclf.HEAD_URL` - **empty until the files are uploaded** (TODO)

Without them, or when the server fails, routing falls back to the keyword rules and the `X-Localllm-Model` header says so:
`gemma4-26b-a4b-qat (th code by embedding p=0.93: best measured)`, `(en math by keyword: ...)`,
`(en general by keyword fallback because no multilingual-e5-small-Q8_0.gguf in ~/.localllm/models ...: ...)`.
The download and the server start run in a background thread, never under the pool lock: the message that kicks them
off waits up to 5 s (a start takes ~1.5 s), others fall back at once (`... task router is downloading ... (not ready
yet)`). A failed or stalled download (30 s read timeout) or a slow start is retried after 5 minutes; a missing file with
no link, a bad head, a server that exits, or 3 failed embeddings in a row (3 s timeout each) turn it off until restart.

## Running the scripts

`data/` (training data, embedding caches) is not committed; `generated.jsonl` is. Paths come from arguments; the server
from `--server`, `$LOCALLLM_LLAMA_SERVER`, the maintainer's Vulkan build if present, else localllm's own llama.cpp.

```
python build_data.py                                   # Hugging Face datasets-server -> data/train.jsonl, heldout.jsonl
python gen_data.py                                     # needs a GPU running gemma-4 -> generated.jsonl
python train_router.py e5-small-Q8_0.gguf --with-generated --export router_head.json
python eval_package.py                                 # uses ~/.localllm/models/{multilingual-e5-small-Q8_0.gguf,router_head.json}
```

CI (`.github/workflows/router.yml`, CPU runner): downloads e5-small, converts it with llama.cpp b11457 (same recipe),
checks it against PyTorch, rebuilds the data with `build_data.py`, trains with the committed `generated.jsonl`, runs
`eval_package.py` on its own head and on the committed `router_head.json`, and uploads the head and the GGUF as
artifacts. It cannot rerun `gen_data.py` (needs a local LLM).
