import itertools
import json
from pathlib import Path


def iter_patents(path, limit=0):
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            patents = (json.loads(line) for line in f if line.strip())
            yield from _fields(patents, limit)
    else:
        with open(path, encoding="utf-8") as f:
            yield from _fields(json.load(f), limit)


def _fields(patents, limit):
    if limit:
        patents = itertools.islice(patents, limit)
    for p in patents:
        yield p.get("ID", ""), p.get("Description", "") or ""
