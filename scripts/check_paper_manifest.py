"""Validate candidate inventory, or require evidence for a paper release."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EVIDENCE = (
    "commit", "resolved_config", "seeds", "dataset_version", "dataset_sha256",
    "split_definition", "checkpoint", "evaluation_command", "metrics",
    "aggregation", "hardware", "runtime_seconds", "peak_memory_gb",
    "figure_or_table_command",
)


def validate(document, root=ROOT, require_complete=False):
    errors = []
    if document.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    entries = document.get("experiments", [])
    if not entries:
        errors.append("experiments must not be empty")
    ids = set()
    for entry in entries:
        name = entry.get("id")
        if not name or name in ids:
            errors.append(f"Missing or duplicate experiment id: {name}")
        ids.add(name)
        for path in [entry.get("launcher", ""), *entry.get("configs", [])]:
            if not path or not (root / path).is_file():
                errors.append(f"{name}: missing file {path!r}")
        status = entry.get("status")
        if status not in {"candidate", "verified"}:
            errors.append(f"{name}: invalid status")
        if require_complete or status == "verified":
            if status != "verified":
                errors.append(f"{name}: not verified")
            if not entry.get("paper_result"):
                errors.append(f"{name}: missing paper table/figure identifier")
            evidence = entry.get("run_evidence") or {}
            for key in REQUIRED_EVIDENCE:
                if key not in evidence or evidence[key] is None or evidence[key] == "" or evidence[key] == [] or evidence[key] == {}:
                    errors.append(f"{name}: missing run evidence: {key}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/paper/manifest.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    errors = validate(json.loads(args.manifest.read_text()), require_complete=args.require_complete)
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print("Manifest passes " + ("release evidence checks." if args.require_complete else "inventory checks; candidates are not reproduced results."))


if __name__ == "__main__":
    main()
