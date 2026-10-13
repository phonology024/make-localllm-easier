#!/usr/bin/env bash
# Measure localllm on a rented cloud GPU (#101): install, doctor, tune, a short eval, report. Run it on a fresh
# Ubuntu 22.04/24.04 machine with an NVIDIA or AMD GPU and its driver already installed (RunPod, Vast.ai, Lambda, ...):
#
#   curl -sSfLO https://raw.githubusercontent.com/phonology024/make-localllm-easier/main/tools/cloud_lab.sh
#   PROVIDER=runpod LANGS=en,th bash cloud_lab.sh
#
# Writes ~/.localllm/report-<date>.json with "host": "cloud:<provider>", so the leaderboard marks the row as a rented
# machine, and prints the prefilled issue link to share it. Takes 30-60 minutes; most of it is the model download and
# the eval. Rough costs per GPU are in issue #101. Nothing here needs root except installing packages.
set -euo pipefail

PROVIDER="${PROVIDER:-unknown}"
LANGS="${LANGS:-en,th}"
LIMIT="${LIMIT:-0}"                       # 0 = full test sets; e.g. 100 for a quick pass
SUITES="${SUITES:-global,regional,math}"
export LOCALLLM_REPORT_HOST="cloud:${PROVIDER}"
export LOCALLLM_YES=1                      # no questions: this runs unattended

if ! command -v vulkaninfo >/dev/null || ! command -v python3 >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3 python3-venv python3-pip libvulkan1 mesa-vulkan-drivers vulkan-tools curl
fi
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" || echo "warning: vulkaninfo sees no GPU. In NVIDIA containers set NVIDIA_DRIVER_CAPABILITIES=all (graphics is off by default)"

python3 -m venv ~/localllm-venv
# shellcheck disable=SC1090
. ~/localllm-venv/bin/activate
pip install -q --upgrade make-localllm-easier
localllm --version

localllm doctor | tee ~/doctor.txt
localllm tune
localllm serve --port 8080 > ~/serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 600); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 1; done
MODEL=$(curl -sf http://127.0.0.1:8080/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
localllm eval --url http://127.0.0.1:8080 --name "$MODEL" --langs "$LANGS" --suites "$SUITES" --limit "$LIMIT"
kill $SERVER
localllm report --no-browser
echo "Done. Open the link above to share the report (you review it before it is posted)."
