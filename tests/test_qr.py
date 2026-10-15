"""QR codes for LAN mode (#122): the standard's worked values, the fixed patterns, and the terminal output."""
import io

import pytest

from localllm import cli, qr


def test_reed_solomon_matches_the_standard_example():
    # "HELLO WORLD", version 1-M: data codewords and the 10 error-correction codewords from the worked example
    data = [32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236, 17, 236, 17]
    assert qr.rs_ecc(data, 10) == [196, 35, 39, 119, 235, 215, 231, 226, 93, 23]


def test_format_and_version_bits():
    assert f"{qr.format_bits('L', 0):015b}" == "111011111000100"
    assert f"{qr.format_bits('M', 0):015b}" == "101010000010010"
    assert f"{qr.format_bits('L', 4):015b}" == "110011000101111"
    assert f"{qr.version_bits(7):018b}" == "000111110010010100"


@pytest.mark.parametrize("text,version", [("x", 1), ("http://192.168.1.23:8080/", 2),
                                          ("http://192.168.100.200:8080/?key=" + "k" * 32, 5)])
def test_fixed_patterns_and_size(text, version):
    m = qr.encode(text, "M")
    n = len(m)
    assert n == 17 + 4 * version and all(len(row) == n for row in m)
    finder = [[max(abs(dx), abs(dy)) != 2 for dx in range(-3, 4)] for dy in range(-3, 4)]
    for cy, cx in ((3, 3), (3, n - 4), (n - 4, 3)):
        assert [row[cx - 3:cx + 4] for row in m[cy - 3:cy + 4]] == finder
    assert [m[6][x] for x in range(8, n - 8)] == [x % 2 == 0 for x in range(8, n - 8)]      # timing patterns
    assert [m[y][6] for y in range(8, n - 8)] == [y % 2 == 0 for y in range(8, n - 8)]
    assert m[n - 8][8]                                                                       # the dark module
    # the two copies of the format bits agree (one around the top-left finder, one split over the other two)
    first = [m[y][8] for y in (0, 1, 2, 3, 4, 5, 7, 8)] + [m[8][x] for x in (7, 5, 4, 3, 2, 1, 0)]
    second = [m[8][n - 1 - i] for i in range(8)] + [m[n - 15 + i][8] for i in range(8, 15)]
    assert first == second and qr.encode(text, "M") == m                                    # and deterministic


def test_too_long_is_refused_and_utf8_fits():
    with pytest.raises(ValueError, match="too long"):
        qr.encode("x" * 200, "M")
    assert len(qr.encode("สวัสดี localllm", "M")) == 29                                        # 26 bytes of UTF-8 -> v3


def test_terminal_blocks_and_plain():
    m = qr.encode("http://10.0.0.5:8080/", "M")
    n = len(m) + 8
    blocks = qr.terminal(m).splitlines()
    assert len(blocks) == (n + 1) // 2 and all(len(line) == n for line in blocks)
    assert set("".join(blocks)) <= set("█▀▄ ") and blocks[0] == "█" * n                     # quiet zone is light
    plain = qr.terminal(m, plain=True).splitlines()
    assert len(plain) == n and all(len(line) == 2 * n for line in plain) and set("".join(plain)) <= set("# ")


class Tty(io.TextIOWrapper):
    def isatty(self):
        return True


def test_cli_prints_plain_on_a_legacy_console_and_nothing_when_piped(monkeypatch, capsys):
    out = Tty(io.BytesIO(), encoding="cp874")
    monkeypatch.setattr("sys.stdout", out)
    cli._print_qr("http://10.0.0.5:8080/")
    out.flush()
    text = out.buffer.getvalue().decode("cp874")
    assert "##" in text and "█" not in text and "?key=" not in text
    monkeypatch.undo()
    cli._print_qr("http://10.0.0.5:8080/")                                                  # pytest's stdout: no tty
    assert capsys.readouterr().out == ""
