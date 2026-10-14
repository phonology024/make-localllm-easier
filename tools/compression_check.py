"""How much smaller would a lossless codec make a GGUF download? (#71: DFloat11 / ZipNN-style, download size only)

  pip install zstandard          # the only extra; not a localllm dependency
  python tools/compression_check.py ~/.localllm/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf

Reads the GGUF's tensor table, samples up to --sample-mb of every tensor type (spread over the file), and compresses
the samples with zstd at --level: as-is, and byte-grouped (the bytes of each 16-bit value split into a high stream -
sign + exponent, the part DFloat11 and ZipNN exploit - and a low stream, compressed separately). The ratio of each type
is then weighted by how many bytes of that type the file holds. One JSON line per file in tools/compression_results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# ggml type id -> (name, block size in elements, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
         12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66),
         17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
         21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56),
         30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17)}
SIXTEEN_BIT = ("F16", "BF16")


def read_tensors(path: Path) -> tuple[list[dict], int]:
    """[{name, type, offset, nbytes}] with absolute offsets, from the GGUF header (v2/v3)."""
    f = open(path, "rb")

    def u(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    def string():
        return f.read(u("Q")).decode("utf-8", "replace")

    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip_value(t):
        if t == 8:
            string()
        elif t == 9:
            et, n = u("I"), u("Q")
            for _ in range(n):
                skip_value(et)
        else:
            f.read(sizes[t])

    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version, n_tensors, n_kv = u("I"), u("Q"), u("Q")
    align = 32
    for _ in range(n_kv):
        key, t = string(), u("I")
        if key == "general.alignment" and t == 4:
            align = u("I")
        else:
            skip_value(t)
    tensors = []
    for _ in range(n_tensors):
        name, nd = string(), u("I")
        dims = [u("Q") for _ in range(nd)]
        ttype, off = u("I"), u("Q")
        n = 1
        for d in dims:
            n *= d
        name_t, blk, bpb = TYPES.get(ttype, (f"type{ttype}", 1, 0))
        tensors.append({"name": name, "type": name_t, "offset": off, "nbytes": n // blk * bpb})
    base = (f.tell() + align - 1) // align * align
    for t in tensors:
        t["offset"] += base
    return tensors, version


def group_bytes(b: bytes) -> tuple[bytes, bytes]:
    return b[1::2], b[0::2]          # little-endian 16-bit: high byte (sign + exponent bits), low byte


def main() -> None:
    import zstandard
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--sample-mb", type=int, default=256, help="per tensor type")
    ap.add_argument("--level", type=int, default=12)
    a = ap.parse_args()
    for path in a.files:
        path = path.expanduser()
        tensors, _ = read_tensors(path)
        by_type: dict[str, list[dict]] = {}
        for t in tensors:
            by_type.setdefault(t["type"], []).append(t)
        rows, t0 = {}, time.time()
        with open(path, "rb") as f:
            for ty, ts in sorted(by_type.items()):
                total = sum(t["nbytes"] for t in ts)
                budget, chunk = a.sample_mb * 2**20, 4 * 2**20
                step = max(1, len(ts) // max(1, budget // chunk))
                sample = bytearray()
                for t in ts[::step]:
                    f.seek(t["offset"])
                    sample += f.read(min(chunk, t["nbytes"]))
                    if len(sample) >= budget:
                        break
                z = zstandard.ZstdCompressor(level=a.level)
                plain = len(z.compress(bytes(sample))) / len(sample)
                grouped = None
                if ty in SIXTEEN_BIT:
                    hi, lo = group_bytes(bytes(sample))
                    grouped = (len(z.compress(hi)) + len(z.compress(lo))) / len(sample)
                rows[ty] = {"gb": round(total / 1e9, 3), "sampled_mb": round(len(sample) / 2**20),
                            "zstd": round(plain, 4), "grouped": round(grouped, 4) if grouped else None}
                print(f"  {ty:8} {total / 1e9:7.2f} GB  zstd {plain:.3f}" +
                      (f"  byte-grouped {grouped:.3f}" if grouped else ""), flush=True)
        size = path.stat().st_size
        best = sum(r["gb"] * 1e9 * min(r["zstd"], r["grouped"] or 9) for r in rows.values())
        covered = sum(r["gb"] * 1e9 for r in rows.values())
        est = (best + size - covered) / size
        rec = {"date": time.strftime("%Y-%m-%d"), "file": path.name, "gb": round(size / 1e9, 2), "level": a.level,
               "types": rows, "estimated_ratio": round(est, 4), "seconds": round(time.time() - t0)}
        with open(ROOT / "tools" / "compression_results.jsonl", "a", encoding="utf-8") as out:
            out.write(json.dumps(rec) + "\n")
        print(f"{path.name}: {size / 1e9:.2f} GB -> ~{size * est / 1e9:.2f} GB ({100 * (1 - est):.1f}% smaller)")


if __name__ == "__main__":
    sys.exit(main())
