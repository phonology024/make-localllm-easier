"""Does the GGUF (llama.cpp, CPU) give the same embeddings as the original PyTorch e5-small?
usage: check_embed.py MODEL.gguf [HF_DIR (default e5-small)] [--server llama-server]
  server: --server, else $LOCALLLM_LLAMA_SERVER, the maintainer's Vulkan build if present, else localllm's own llama.cpp"""
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from localllm import runtime, taskclf  # noqa: E402

TEXTS = ["query: How many moons does Jupiter have?", "query: ช่วยแปลประโยคนี้เป็นภาษาอังกฤษให้หน่อย",
         "query: 请帮我写一个Python函数计算斐波那契数列", "query: Janet has 16 eggs and eats 3. How many are left?",
         "query: Écris un poème sur la mer.", "query: इस लेख का सारांश लिखिए"]
WIN_SERVER = r"C:\Users\user\tools\llama.cpp-b11457\vulkan\llama-server.exe"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model"); ap.add_argument("hf", nargs="?", default="e5-small"); ap.add_argument("--server")
A = ap.parse_args()
SERVER = A.server or os.environ.get("LOCALLLM_LLAMA_SERVER") or (WIN_SERVER if Path(WIN_SERVER).exists() else None) \
    or str(runtime.find_server())
PORT = taskclf._free_port()

tok = AutoTokenizer.from_pretrained(A.hf)
model = AutoModel.from_pretrained(A.hf).eval()
with torch.no_grad():
    b = tok(TEXTS, padding=True, return_tensors="pt")
    h = model(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    ref = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)

srv = subprocess.Popen([SERVER, "-m", A.model, "--embedding", "--pooling", "mean", "-ngl", "0", "--port", str(PORT),
                        "-c", "512", "-b", "512", "-ub", "512"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); break
        except OSError:
            time.sleep(0.2)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/embeddings", data=json.dumps({"input": TEXTS}).encode(),
                                 headers={"Content-Type": "application/json"})
    got = torch.tensor([d["embedding"] for d in json.load(urllib.request.urlopen(req))["data"]])
    got = torch.nn.functional.normalize(got, dim=-1)
    cos = (got * ref).sum(-1)
    print("cosine vs PyTorch per text:", [round(float(c), 4) for c in cos])
finally:
    srv.terminate()
