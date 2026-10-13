"""#97: the savings meter counts local answers from llama-server's own token counts; dollars only at the user's prices."""
import json

from localllm import gateway, meter
from test_gateway import _post, _up


def test_parse_usage_from_every_shape():
    # OpenAI non-stream: usage is the whole prompt, timings split off what came from the cache
    body = {"usage": {"prompt_tokens": 1200, "completion_tokens": 80}, "timings": {"prompt_n": 200, "cache_n": 1000,
                                                                                    "predicted_n": 80}}
    assert meter.parse_usage(json.dumps(body).encode()) == {"input": 1200, "output": 80, "cached": 1000}
    # a stream's last chunk with timings only
    sse = b'data: {"choices":[],"timings":{"prompt_n":50,"cache_n":950,"predicted_n":12}}\n\ndata: [DONE]\n\n'
    assert meter.parse_usage(sse) == {"input": 1000, "output": 12, "cached": 950}
    # Anthropic
    msg = b'event: message_delta\ndata: {"usage":{"input_tokens":30,"output_tokens":7}}\n\n'
    assert meter.parse_usage(msg) == {"input": 30, "output": 7, "cached": 0}
    assert meter.parse_usage(b'{"error":"x"}') is None


def test_month_totals_and_prices(tmp_path):
    m = meter.Meter(tmp_path / "s.json")
    assert meter.line(m.month()) == "nothing answered locally yet this month"
    m.add({"input": 1_000_000, "output": 100_000, "cached": 750_000})
    m.add({"input": 1_000_000, "output": 100_000, "cached": 750_000})
    got = m.month()
    assert got["requests"] == 2 and "saved_usd" not in got
    assert "75% of prompt tokens from the cache" in meter.line(got) and "--price" in meter.line(got)
    m.set_price(3, 15, "my cloud model")
    assert m.month()["saved_usd"] == 9.0                      # 2M x $3 + 0.2M x $15
    assert "$9.00 at your prices (my cloud model)" in meter.line(m.month())


def test_gateway_counts_local_answers_only(tmp_path):
    up, uurl = _up()
    m = meter.Meter(tmp_path / "s.json")
    g = gateway.serve(uurl, port=0, model_name="t", meter=m)
    base = f"http://127.0.0.1:{g.server_address[1]}"
    _post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}).read()
    _post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True}).read()
    _post(base + "/api/chat", {"messages": [{"role": "user", "content": "hi"}], "stream": False}).read()
    got = m.month()
    assert got["requests"] == 3 and got["output"] == 2 + 2 + 2 and got["input"] == 3
    _post(base + "/v1/embeddings", {"input": "x"}).read()       # not an answer: not counted
    assert m.month()["requests"] == 3
    g.shutdown(); up.shutdown()
