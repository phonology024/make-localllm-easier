# localllm-harness (experimental add-on, #88)

A plan / tool / verify / retry loop for small local models. It is its own package: the base
`pip install make-localllm-easier` never installs or imports it. Pros and cons: [docs/addons.md](../../docs/addons.md).

```
pip install ./addons/harness            # from a clone; not on PyPI until #88's measurements say it pays off
localllm serve --profile max            # long agent prompts fit
localllm-agent "make the failing test pass" --check "python -m unittest" --project path/to/project
```

- **Plan:** one call writes a short plan; *done* is fixed before any work: the `--check` command passes.
- **Work:** tools `list_files`, `read_file`, `write_file`, `run`. Files change only through `write_file`, inside the
  project; a path outside it is refused unless you answer yes. `run` and the check execute in localllm's Docker sandbox
  (no network, 1 GB / 2 CPUs / 256 processes, non-root) on a throwaway copy of the project, so a command never touches
  your files or the network. Without Docker, nothing is run.
- **Verify and retry:** a failed check starts the next round from a fresh context with the check's output (a small
  context keeps the KV cache and RAM small), up to `--rounds` (default 3).

Status: tested with fake models only. Before it ships into the base install it has to beat the bare model and
OpenCode/Aider on the same local model on checkable tasks (#88, measured by the maintainers on the GPU).
