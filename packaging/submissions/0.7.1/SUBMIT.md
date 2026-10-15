# Submitting 0.7.1 to winget and Homebrew (maintainer only)

The files next to this note are ready. They come from the v0.7.1 release workflow, with one fix: the Homebrew formula
now carries the hash of the tarball PyPI actually serves (`fb2017…`). The workflow's first formula had the hash of a
tarball it had built again itself (`d687a4…`), and `brew install` would have refused it. Hashes checked on 2026-10-15:

| File | SHA-256 | Matches |
|---|---|---|
| `localllm-windows-x64.exe` | `e260b768…5f413` | `SHA256SUMS` on the v0.7.1 release |
| `make_localllm_easier-0.7.1.tar.gz` | `fb201722…c4ec8` | PyPI's listing for 0.7.1 |

Both steps publish under your GitHub account, outside this repo. Run them only when you want the package to be public.

## winget (Windows, PowerShell or Git Bash, from the folder that holds your make-localllm-easier clone)

```
gh repo fork microsoft/winget-pkgs --clone=false
git clone --filter=blob:none --sparse https://github.com/phonology024/winget-pkgs.git
cd winget-pkgs
git sparse-checkout set manifests/p/phonology024
git checkout -b localllm-0.7.1
mkdir -p manifests/p/phonology024/localllm/0.7.1
cp ../make-localllm-easier/packaging/submissions/0.7.1/winget/*.yaml manifests/p/phonology024/localllm/0.7.1/
winget validate --manifest manifests/p/phonology024/localllm/0.7.1
git add manifests/p/phonology024/localllm/0.7.1
git commit -m "New package: phonology024.localllm version 0.7.1"
git push -u origin localllm-0.7.1
gh pr create -R microsoft/winget-pkgs --head phonology024:localllm-0.7.1 --title "New package: phonology024.localllm version 0.7.1" --body "Portable single-file build of localllm (MIT), from https://github.com/phonology024/make-localllm-easier/releases/tag/v0.7.1"
```

Optional local test before the PR (admin PowerShell once: `winget settings --enable LocalManifestFiles`):
`winget install --manifest manifests/p/phonology024/localllm/0.7.1`.
The winget bots validate the PR and may ask for changes; after it merges: `winget install localllm`.

## Homebrew tap (any machine with gh; testing needs a Mac)

```
gh repo create phonology024/homebrew-tap --public --description "Homebrew tap for localllm" --clone
cd homebrew-tap
mkdir -p Formula
cp ../make-localllm-easier/packaging/submissions/0.7.1/homebrew/localllm.rb Formula/localllm.rb
git add Formula/localllm.rb
git commit -m "localllm 0.7.1"
git push -u origin HEAD:main
```

On a Mac: `brew install phonology024/tap/localllm && brew test localllm`. From the next release on, copy the formula
from the release workflow's `manifests` artifact; it now takes the hash from PyPI.
