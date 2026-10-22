import collections
import json

import pandas as pd

from .headers import GROUPS
from .sections import chunk_spans, crossing, measure, section_spans


class Collector:
    def __init__(self, out, sent_per_chunk, overlap):
        self.out = out
        self.sent_per_chunk = sent_per_chunk
        self.overlap = overlap
        self.n = 0
        self.h_docs, self.h_occ, self.h_example, self.h_meta = collections.Counter(), collections.Counter(), {}, {}
        self.orders = collections.Counter()
        self.n_types = collections.Counter()
        self.no_headers = []
        self.lengths = []
        self.n_chunks = 0
        self.n_cut_chunks = 0
        self.n_patents_cut = 0
        self.per_patent = open(out / "per_patent_headers.jsonl", "w", encoding="utf-8")

    def add(self, patent_id, text, headers):
        self.n += 1
        for h in headers:
            self.h_occ[h["header"]] += 1
            self.h_example.setdefault(h["header"], patent_id)
            self.h_meta[h["header"]] = (h["group"], h["source"])
        for header in {h["header"] for h in headers}:
            self.h_docs[header] += 1

        groups = list(dict.fromkeys(h["group"] for h in headers if h["group"] != "other"))
        self.orders[" > ".join(groups)] += 1
        self.n_types[len(groups)] += 1
        if not groups:
            self.no_headers.append({"patent_id": patent_id, "description_start": text[:300]})

        json.dump({"patent_id": patent_id, "headers": headers}, self.per_patent, ensure_ascii=False)
        self.per_patent.write("\n")

        if not headers or not text:
            return
        spans = section_spans(headers, len(text))
        per_group = collections.defaultdict(collections.Counter)
        for group, a, e in spans:
            per_group[group].update(measure(text[a:e]))
        for group, m in per_group.items():
            self.lengths.append({"patent_id": patent_id, "group": group, **m, "share": m["chars"] / len(text)})

        chunks = chunk_spans(text, self.sent_per_chunk, self.overlap)
        cut = crossing(chunks, [a for _, a, _ in spans[1:]])
        self.n_chunks += len(chunks)
        self.n_cut_chunks += sum(cut)
        self.n_patents_cut += any(cut)

    def save(self, vocab_size, mapping_size):
        self.per_patent.close()
        n = self.n

        pd.DataFrame([
            {"header": h, "group": self.h_meta[h][0], "source": self.h_meta[h][1], "n_patents": c,
             "share_patents": round(c / n, 4), "n_occurrences": self.h_occ[h], "example_patent": self.h_example[h]}
            for h, c in self.h_docs.items()
        ]).sort_values("n_patents", ascending=False).to_csv(self.out / "headers.csv", index=False)

        pd.DataFrame([
            {"order": o or "(нет)", "n_patents": c, "share_patents": round(c / n, 4)}
            for o, c in self.orders.most_common()
        ]).to_csv(self.out / "orders.csv", index=False)

        pd.DataFrame(self.no_headers, columns=["patent_id", "description_start"]).to_csv(
            self.out / "no_headers.csv", index=False)

        sections = section_table(pd.DataFrame(self.lengths), n)
        sections.to_csv(self.out / "sections.csv", index=False)

        n_with = n - self.n_types[0]
        weighted = sum(k * c for k, c in self.n_types.items() if k)
        summary = {
            "n_patents": n,
            "vocab_size": vocab_size,
            "mapping_size": mapping_size,
            "n_section_types": len(GROUPS),
            "patents_with_sections": n_with,
            "share_with_sections": round(n_with / n, 4),
            "section_types_per_patent_mean": round(weighted / n_with, 2) if n_with else 0,
            "section_types_per_patent": {str(k): c for k, c in sorted(self.n_types.items())},
            "sentences_per_chunk": self.sent_per_chunk,
            "chunk_overlap": self.overlap,
            "n_chunks": self.n_chunks,
            "share_chunks_crossing_boundary": round(self.n_cut_chunks / max(self.n_chunks, 1), 4),
            "share_patents_with_crossing": round(self.n_patents_cut / max(n_with, 1), 4),
        }
        (self.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary, sections


def section_table(lengths, n_patents):
    rows = []
    for group, g in lengths.groupby("group"):
        row = {"group": group, "n_patents": len(g), "share_patents": round(len(g) / n_patents, 4),
               "mean_share": round(g["share"].mean(), 4)}
        for col in ["chars", "words", "sentences"]:
            row[f"{col}_mean"] = round(g[col].mean())
            row[f"{col}_median"] = round(g[col].median())
            row[f"{col}_q10"] = round(g[col].quantile(0.1))
            row[f"{col}_q90"] = round(g[col].quantile(0.9))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("chars_mean", ascending=False)
