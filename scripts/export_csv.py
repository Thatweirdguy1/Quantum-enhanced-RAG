"""Export recorded per-query experiment metrics and retrieval traces to CSV.

    python -m scripts.export_csv results/experiment.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def export_csv(source: Path, target: Path) -> int:
    data = json.loads(source.read_text(encoding="utf8"))
    clean = data["clean"]
    metrics = sorted({key for block in clean.values()
                      for row in block["per_query"].values() for key in row})
    columns = ["arm", "system", "query_id", "config_hash", "seed", "latency_ms",
               "ranked_doc_ids", "scores", *metrics]
    count = 0
    with target.open("w", encoding="utf8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        arms = (("clean", clean),
                ("poisoned", data.get("poisoned", {}).get("systems", {})))
        for arm, systems in arms:
            for system, block in systems.items():
                details = block.get("query_details", {})
                for query_id, values in block.get("per_query", {}).items():
                    detail = details.get(query_id, {})
                    writer.writerow({
                        "arm": arm, "system": system, "query_id": query_id,
                        "config_hash": data["provenance"]["config_hash"],
                        "seed": data["provenance"]["seed"],
                        "latency_ms": detail.get("latency_ms", ""),
                        "ranked_doc_ids": json.dumps(detail.get("ranked_doc_ids", [])),
                        "scores": json.dumps(detail.get("scores", [])),
                        **values,
                    })
                    count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.source.with_suffix(".csv")
    print(f"wrote {export_csv(args.source, output)} rows to {output}")
