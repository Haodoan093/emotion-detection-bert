from __future__ import annotations
import json
from pathlib import Path

bench = Path("reports/emotion/benchmark_20260423")
rows = []
for p in sorted(bench.glob("eval_*.json")):
    obj = json.loads(p.read_text(encoding="utf-8"))
    metrics = obj.get("metrics", {})
    rows.append({
        "file": p.name,
        "model": obj.get("model"),
        "samples": obj.get("samples"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
    })

out = bench / "summary_table.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved={out}")
for r in rows:
    print(f"{r['file']}: acc={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")
