# Embedding task router (issues #28, #35) - research scripts

Goal: classify each message as `general` / `math` / `code` / `translate` in any language, in < 20 ms, with a model far
smaller than Laya (614 MB), then let the measured catalog scores pick the model.

| file | what |
|---|---|
| `check_embed.py` | llama.cpp GGUF embeddings vs PyTorch (cosine 0.992-0.9998 for e5-small Q8_0) |
| `build_data.py` | training data labelled by source (MGSM, GSM8K, MBPP, CRUXEval, FLORES, Aya, oasst2, Global-MMLU, ThaiExam) |
| `gen_data.py` | short everyday-style requests per class in 15 languages, written by a local LLM |
| `train_router.py` | embed with llama-server `--embedding --pooling mean`, logistic-regression head, eval vs keyword rules, `--export` the head |
| `eval_package.py` | the router exactly as localllm runs it (`localllm/taskclf.py`): accuracy per order, latency, gateway reasons |
| `eval_laya.py` | Laya (multilingual checkpoint, zero-shot choice) on the same test set |
| `shared_test.jsonl` | 144 hand-written requests, 15 languages, never used for training |

Model: `intfloat/multilingual-e5-small` (MIT) converted with `convert_hf_to_gguf.py` after setting
`architectures: ["XLMRobertaModel"]` and removing `pad_token_id` (BERT body + XLM-R sentencepiece tokenizer; keeping
pad_token_id would shift position embeddings by one). Q8_0 = 126 MB. Inputs need the `query: ` prefix, cap ~450 chars.

First results (RX 9070 XT PC, CPU, shared test): Laya 77.8% (103 ms median) | embedding head 64.6% | embedding head with
keyword rules first 72.2% (~6 ms) | keyword rules alone 56.9%. Held-out (same style as training) 98.6%: the gap is
training-data style, being fixed with generated short requests.

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

## Running the scripts

`data/` (training data, embedding caches) is not committed. Paths come from arguments; the server from `--server`,
`$LOCALLLM_LLAMA_SERVER`, the maintainer's Vulkan build if present, else localllm's own llama.cpp.

```
python build_data.py                                   # Hugging Face datasets-server -> data/train.jsonl, heldout.jsonl
python gen_data.py                                     # needs a GPU running gemma-4 -> data/generated.jsonl
python train_router.py e5-small-Q8_0.gguf --extra data/generated.jsonl --export router_head.json
python eval_package.py                                 # uses ~/.localllm/models/{multilingual-e5-small-Q8_0.gguf,router_head.json}
```

CI (`.github/workflows/router.yml`, CPU runner): downloads e5-small, converts it with llama.cpp b11457 (same recipe),
checks it against PyTorch, rebuilds the data with `build_data.py`, trains, runs `eval_package.py`, and uploads the head
and the GGUF as artifacts. It cannot run `gen_data.py` (needs a local LLM), so its head is trained **without** the
generated short requests.
