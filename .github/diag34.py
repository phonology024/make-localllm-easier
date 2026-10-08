# TEMPORARY diagnostics for issue #34 (removed before the PR): why the canonical find_zero (HumanEval/32) misses
# |poly(x)| <= 1e-4 on some HumanEval+ inputs. Runs the dataset's own reference code on a throwaway CI runner.
import ast
from localllm import bench

r = next(r for r in bench._rows("evalplus/humanevalplus", "default") if r["task_id"] == "HumanEval/32")
print(r["prompt"] + r["canonical_solution"])
tree = ast.parse(r["test"])
check = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check")
vals = {t.targets[0].id: ast.literal_eval(t.value) for t in check.body if isinstance(t, ast.Assign)}
inputs, results = vals["inputs"], vals["results"]
g = {}
exec(r["prompt"] + r["canonical_solution"], g)
bad = []
for inp, exp in zip(inputs, results):
    out = g["find_zero"](*inp)
    y = g["poly"](inp[0], out)
    if abs(y) > 1e-4:
        bad.append((inp[0], out, exp, y))
print(f"{len(inputs)} inputs, {len(bad)} with |poly(out)| > 1e-4")
for xs, out, exp, y in bad:
    print(f"  xs={xs} out={out!r} exp={exp!r} poly(out)={y:.3g} same_as_exp={out == exp}")
