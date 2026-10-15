"""QR codes for the terminal, standard library only (#122): `localllm serve --host 0.0.0.0` prints one with the URL.

Byte mode, error correction L or M, versions 1-7 (up to 154 bytes at L): plenty for a LAN URL. Follows ISO/IEC 18004
(the layout as in Nayuki's reference QR Code generator); tests check the Reed-Solomon, format and version bits against
the standard's worked values.
"""
from __future__ import annotations

# per version 1..7: (EC codewords per block, number of blocks) for L and M; total codewords per version
ECC = {"L": [(7, 1), (10, 1), (15, 1), (20, 1), (26, 1), (18, 2), (20, 2)],
       "M": [(10, 1), (16, 1), (26, 1), (18, 2), (24, 2), (16, 4), (18, 4)]}
TOTAL = [26, 44, 70, 100, 134, 172, 196]
ALIGN = [[], [6, 18], [6, 22], [6, 26], [6, 30], [6, 34], [6, 22, 38]]
FORMAT_ECL = {"L": 1, "M": 0}


def _mul(x: int, y: int) -> int:
    z = 0
    for i in reversed(range(8)):
        z = (z << 1) ^ ((z >> 7) * 0x11D)
        z ^= ((y >> i) & 1) * x
    return z


def rs_ecc(data: list[int], degree: int) -> list[int]:
    """Reed-Solomon error-correction codewords over GF(256), polynomial 0x11D."""
    div, root = [0] * (degree - 1) + [1], 1
    for _ in range(degree):
        for j in range(degree):
            div[j] = _mul(div[j], root)
            if j + 1 < degree:
                div[j] ^= div[j + 1]
        root = _mul(root, 0x02)
    rem = [0] * degree
    for b in data:
        factor = b ^ rem.pop(0)
        rem.append(0)
        for i, coef in enumerate(div):
            rem[i] ^= _mul(coef, factor)
    return rem


def format_bits(ecl: str, mask: int) -> int:
    data = FORMAT_ECL[ecl] << 3 | mask
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    return (data << 10 | rem) ^ 0x5412


def version_bits(version: int) -> int:
    rem = version
    for _ in range(12):
        rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
    return version << 12 | rem


def _codewords(data: bytes, version: int, ecl: str) -> list[int]:
    ec_len, blocks = ECC[ecl][version - 1]
    capacity = TOTAL[version - 1] - ec_len * blocks
    bits = [0, 1, 0, 0] + [int(b) for b in f"{len(data):08b}"] + [int(b) for byte in data for b in f"{byte:08b}"]
    bits += [0] * min(4, capacity * 8 - len(bits))
    bits += [0] * (-len(bits) % 8)
    words = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = 0xEC
    while len(words) < capacity:
        words.append(pad)
        pad ^= 0xEC ^ 0x11
    short = blocks - TOTAL[version - 1] % blocks
    short_len = TOTAL[version - 1] // blocks
    split, k = [], 0
    for i in range(blocks):
        n = short_len - ec_len + (0 if i < short else 1)
        block = words[k:k + n]
        k += n
        split.append((block, rs_ecc(block, ec_len)))
    out = []
    for i in range(max(len(b) for b, _ in split)):
        out += [b[i] for b, _ in split if i < len(b)]
    for i in range(ec_len):
        out += [e[i] for _, e in split]
    return out


def _penalty(m: list[list[bool]]) -> int:
    n, score = len(m), 0
    lines = m + [list(c) for c in zip(*m)]
    for line in lines:                                     # N1: runs of 5+
        run, prev = 0, None
        for v in line:
            run = run + 1 if v == prev else 1
            prev = v
            if run == 5:
                score += 3
            elif run > 5:
                score += 1
        s = "".join("1" if v else "0" for v in line)       # N3: finder-like patterns
        score += 40 * (s.count("10111010000") + s.count("00001011101"))
    for y in range(n - 1):                                 # N2: 2x2 blocks
        for x in range(n - 1):
            if m[y][x] == m[y][x + 1] == m[y + 1][x] == m[y + 1][x + 1]:
                score += 3
    dark = sum(map(sum, m))                                # N4: balance
    score += (abs(dark * 20 - n * n * 10) + n * n - 1) // (n * n) * 10 - 10 if dark * 20 != n * n * 10 else 0
    return score


MASKS = [lambda y, x: (x + y) % 2 == 0, lambda y, x: y % 2 == 0, lambda y, x: x % 3 == 0,
         lambda y, x: (x + y) % 3 == 0, lambda y, x: (x // 3 + y // 2) % 2 == 0,
         lambda y, x: x * y % 2 + x * y % 3 == 0, lambda y, x: (x * y % 2 + x * y % 3) % 2 == 0,
         lambda y, x: ((x + y) % 2 + x * y % 3) % 2 == 0]


def encode(text: str, ecl: str = "M") -> list[list[bool]]:
    """The QR matrix (True = dark) for `text`, in the smallest version 1-7 that holds it."""
    data = text.encode("utf-8")
    for version in range(1, 8):
        ec_len, blocks = ECC[ecl][version - 1]
        if 4 + 8 + 8 * len(data) <= (TOTAL[version - 1] - ec_len * blocks) * 8:
            break
    else:
        raise ValueError(f"too long for a version 1-7 QR code at level {ecl}: {len(data)} bytes")
    size = 17 + 4 * version
    mod = [[False] * size for _ in range(size)]
    fn = [[False] * size for _ in range(size)]

    def put(x: int, y: int, dark: bool) -> None:
        mod[y][x], fn[y][x] = dark, True

    for i in range(size):                                  # timing patterns
        put(6, i, i % 2 == 0)
        put(i, 6, i % 2 == 0)
    for cx, cy in ((3, 3), (size - 4, 3), (3, size - 4)):  # finders with their separators
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < size and 0 <= y < size:
                    d = max(abs(dx), abs(dy))
                    put(x, y, d not in (2, 4))
    pos = ALIGN[version - 1]
    for i, ay in enumerate(pos):                           # alignment patterns
        for j, ax in enumerate(pos):
            if (i, j) in ((0, 0), (0, len(pos) - 1), (len(pos) - 1, 0)):
                continue
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    put(ax + dx, ay + dy, max(abs(dx), abs(dy)) != 1)

    def draw_format(mask: int) -> None:
        bits = format_bits(ecl, mask)
        bit = lambda i: (bits >> i) & 1 == 1               # noqa: E731
        for i in range(6):
            put(8, i, bit(i))
        put(8, 7, bit(6)); put(8, 8, bit(7)); put(7, 8, bit(8))
        for i in range(9, 15):
            put(14 - i, 8, bit(i))
        for i in range(8):
            put(size - 1 - i, 8, bit(i))
        for i in range(8, 15):
            put(8, size - 15 + i, bit(i))
        put(8, size - 8, True)                             # the dark module

    draw_format(0)                                         # reserve the format areas
    if version >= 7:
        vb = version_bits(version)
        for i in range(18):
            a, b, v = size - 11 + i % 3, i // 3, (vb >> i) & 1 == 1
            put(a, b, v)
            put(b, a, v)
    words = _codewords(data, version, ecl)
    i, right = 0, size - 1
    while right >= 1:                                      # zigzag placement, two columns at a time
        if right == 6:
            right = 5
        upward = (right + 1) & 2 == 0
        for vert in range(size):
            y = size - 1 - vert if upward else vert
            for x in (right, right - 1):
                if not fn[y][x] and i < len(words) * 8:
                    mod[y][x] = (words[i >> 3] >> (7 - (i & 7))) & 1 == 1
                    i += 1
        right -= 2
    best = None
    for mask in range(8):
        trial = [[mod[y][x] ^ (not fn[y][x] and MASKS[mask](y, x)) for x in range(size)] for y in range(size)]
        saved = [row[:] for row in mod]
        mod[:] = trial
        draw_format(mask)
        score = _penalty(mod)
        if best is None or score < best[0]:
            best = (score, [row[:] for row in mod])
        mod[:] = saved
    return best[1]


def terminal(matrix: list[list[bool]], quiet: int = 4, plain: bool = False) -> str:
    """Two rows per text line with half blocks; light modules are drawn, so it reads right on a dark terminal.
    plain: "##" per light module, one line per row, for consoles that can't print block characters."""
    n = len(matrix)
    light = lambda y, x: not (0 <= y < n and 0 <= x < n and matrix[y][x])   # noqa: E731 - quiet zone is light
    if plain:
        return "\n".join("".join("##" if light(y, x) else "  " for x in range(-quiet, n + quiet))
                         for y in range(-quiet, n + quiet))
    lines = []
    for y in range(-quiet, n + quiet, 2):
        lines.append("".join({(True, True): "█", (True, False): "▀", (False, True): "▄", (False, False): " "}
                             [(light(y, x), light(y + 1, x))] for x in range(-quiet, n + quiet)))
    return "\n".join(lines)
