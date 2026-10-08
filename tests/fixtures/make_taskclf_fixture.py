"""Writes taskclf_fixture.json: a small router head (384 dims like e5-small, fitted by sklearn on synthetic clusters),
3 raw embeddings and the labels/probabilities numpy computes for them. tests/test_taskclf.py checks the package's
pure-Python head math against it. Dev-only (numpy, scikit-learn): python tests/fixtures/make_taskclf_fixture.py"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

LABELS = ["general", "math", "code", "translate"]
TEXTS = ["What's a good name for a grey cat?", "เขียนฟังก์ชันเรียงลำดับตัวเลขให้หน่อย", "Comment dit-on bonjour en japonais ?"]
WANT = ["general", "code", "translate"]     # keyword rules say "general" for all three: the embedding decides

rng = np.random.default_rng(35)
centers = rng.normal(size=(4, 384))
x = np.concatenate([c + 2.5 * rng.normal(size=(200, 384)) for c in centers])
y = np.repeat(LABELS, 200)
x /= np.linalg.norm(x, axis=1, keepdims=True)
clf = LogisticRegression(C=2, max_iter=3000, class_weight="balanced").fit(x, y)

head = {"labels": [str(c) for c in clf.classes_], "coef": np.round(clf.coef_, 6).tolist(),
        "intercept": np.round(clf.intercept_, 6).tolist(), "prefix": "query: ", "max_chars": 450}
emb = np.round(np.array([centers[LABELS.index(w)] + 3.0 * rng.normal(size=384) for w in WANT]) * [[1], [7.5], [0.2]], 6)

# what the package must reproduce: softmax(W x/|x| + b) in float64 on the rounded numbers
W, b = np.array(head["coef"]), np.array(head["intercept"])
z = (emb / np.linalg.norm(emb, axis=1, keepdims=True)) @ W.T + b
p = np.exp(z - z.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
labels = [head["labels"][i] for i in p.argmax(1)]
assert labels == WANT, labels
assert np.allclose(p, clf.predict_proba(emb / np.linalg.norm(emb, axis=1, keepdims=True)), atol=1e-4)   # = sklearn

cases = [{"text": t, "embedding": e.tolist(), "label": l, "p": float(q.max()), "probs": q.tolist()}
         for t, e, l, q in zip(TEXTS, emb, labels, p)]
out = Path(__file__).with_name("taskclf_fixture.json")
out.write_text(json.dumps({"head": head, "cases": cases}, ensure_ascii=False) + "\n", encoding="utf-8")
print(out, out.stat().st_size, "bytes", [(c["label"], round(c["p"], 3)) for c in cases])
