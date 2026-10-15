# Installing without Python (#46)

Every `v*` tag runs `.github/workflows/release.yml`: PyInstaller builds a single `localllm` executable on Windows,
Linux and macOS (arm64 and Intel), each one is smoke-tested (`--version`, `list`, `connect cline`), Linux also gets an
AppImage, and all of them go onto the GitHub release with `SHA256SUMS`. The same job writes the winget manifests and the
Homebrew formula from those exact files (artifact `manifests`).

Build one locally: `python -m pip install pyinstaller==6.11.1 && python packaging/build_exe.py` (Windows: 7.8 MB).

The executable still downloads llama.cpp and the model on first use, with the same SHA-256 checks. On the first run
it shows `doctor` and asks before the multi-GB model download (`--yes` or `LOCALLLM_YES=1` skips the question).

## Publishing a release (maintainer, once per version)

These steps use your own accounts, so the workflow doesn't do them:

(Ready-to-run commands for 0.7.1, with the checked files: [submissions/0.7.1/SUBMIT.md](submissions/0.7.1/SUBMIT.md).)

1. **PyPI**: `publish.yml` already runs on the tag (Trusted Publishing).
2. **winget**: fork [microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs), copy
   `manifests/p/phonology024/localllm/<version>/` from the `manifests` artifact into the same path, open a PR. The bots
   validate it (`winget validate --manifest <folder>` locally first). After it merges: `winget install localllm`.
3. **Homebrew**: create the repo `phonology024/homebrew-tap` once, then copy `homebrew/localllm.rb` to
   `Formula/localllm.rb` in it. Users: `brew install phonology024/tap/localllm`. (homebrew-core wants more stars and
   history first.)
4. **AppImage**: nothing to do; `localllm-linux-x64.AppImage` is on the release. `chmod +x` and run it.

Unsigned binaries: Windows SmartScreen and macOS Gatekeeper warn on first start (macOS: right-click > Open, or
`xattr -d com.apple.quarantine localllm-macos-arm64`). Code signing needs paid certificates; winget and Homebrew are
the routes that avoid the warning for most people.
