# TEMPORARY diagnostics for issue #34 (removed before the PR): does the canonical find_zero (HumanEval/32) reproduce
# the recorded answers when poly() sums floats the pre-3.12 way (Python 3.12 made sum() of floats compensated)?
import ast, functools, math, operator, sys
from localllm import bench

print(sys.version)
r = next(r for r in bench._rows("evalplus/humanevalplus", "default") if r["task_id"] == "HumanEval/32")
check = next(n for n in ast.parse(r["test"]).body if isinstance(n, ast.FunctionDef) and n.name == "check")
vals = {t.targets[0].id: ast.literal_eval(t.value) for t in check.body if isinstance(t, ast.Assign)}
inputs, results = vals["inputs"], vals["results"]
for label, total in (("sum() of this Python", sum), ("naive left-to-right +", lambda v: functools.reduce(operator.add, v, 0))):
    g = {"math": math, "total": total}
    src = (r["prompt"] + r["canonical_solution"]).replace("return sum([coeff", "return total([coeff")
    exec(src, g)
    same = bad = bad_naive_oracle = 0
    for (xs,), exp in zip(inputs, results):
        out = g["find_zero"](xs)
        same += out == exp
        ok = abs(g["poly"](xs, out)) <= 1e-4 or out == exp
        bad += not ok
    print(f"{label:24} out == recorded answer for {same}/{len(inputs)}; fails |poly(out)|<=1e-4-or-recorded: {bad}")
for (xs,), exp in zip(inputs, results):
    y = sum(c * math.pow(exp, i) for i, c in enumerate(xs))
    if abs(y) > 1e-4:
        print(f"  recorded answer is not a 1e-4 root: xs={xs} exp={exp} poly(exp)={y:.3g}")
