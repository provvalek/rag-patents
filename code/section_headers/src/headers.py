import collections
import re

import pandas as pd

from .data import iter_patents

CAPS_WORD = r"(?!FIGS?\b)[A-Z][A-Z0-9\-/&,()'.]*[A-Z0-9)]"
CAPS_RUN = rf"{CAPS_WORD}(?:\s+{CAPS_WORD})*"
SECTION_START = r"\s+(?:[A-Z][a-z]|[AI]\s|FIGS?\.|\(?\d{1,2}[.)]\s|\([a-z]\)\s)"

STRICT_RE = re.compile(rf"(?:^|(?<=[.:;)\]]))\s*({CAPS_RUN})(?={SECTION_START})")
LOOSE_RE = re.compile(rf"(?<![A-Za-z0-9\-])({CAPS_RUN})(?={SECTION_START}|\s*$)")
LEADING_JUNK_RE = re.compile(r"\(?[A-Z]{1,3}[.)]")
MIN_LETTERS = 4

GROUP_RULES = [
    ("cross_reference", r"CROSS|RELATED APPLICATION|PRIOR APPLICATION|PRIORITY|FEDERALLY|GOVERNMENT|INCORPORATION BY REFERENCE"),
    ("drawings", r"DRAWING|FIGURE|ILLUSTRATIONS|DIAGRAMS"),
    ("detailed", r"DETAILED DESCRIPTION|EMBODIMENT|MODES? FOR|MODE\(S\) FOR|MODES OF THE INVENTION|BEST MODE"
                 r"|DESCRIPTION OF THE INVENTION"),
    ("background", r"BACKGROUND|PRIOR[- ]ART|RELATED[- ]ART|STATE OF THE ART|CITATIONS? LIST|PATENT LITERATURE"
                   r"|PATENT DOCUMENT|RELATED DOCUMENT"),
    ("field", r"FIELD"),
    ("summary", r"SUMMARY|DISCLOSURE|OBJECT(?:IVE)?S? OF THE INVENTION|EFFECTS? OF (?:THE )?INVENTION"
                r"|STATEMENT OF (?:THE )?INVENTION|CONTENTS OF THE INVENTION"),
    ("examples", r"EXAMPLE|EXPERIMENTAL|EXPERIMENTS|EXEMPLIFICATION|TEST METHODS"),
    ("industrial", r"INDUSTRIAL APPLICAB|INDUSTRIAL APPLICATION|REFERENCE SIGNS|REFERENCE NUMERALS?|REFERENCE CHARACTERS"),
]
GROUP_RES = [(g, re.compile(rf"\b(?:{p})")) for g, p in GROUP_RULES]
GROUPS = [g for g, _ in GROUP_RULES]


def group_by_keywords(header):
    for g, rx in GROUP_RES:
        if rx.search(header):
            return g
    return "other"


def is_header_candidate(text):
    return sum(c.isalpha() for c in text) >= MIN_LETTERS and not re.search(r"\d", text)


def build_vocab(path, min_docs, limit=0):
    docs = collections.Counter()
    for _, text in iter_patents(path, limit):
        docs.update({m.group(1) for m in STRICT_RE.finditer(text) if is_header_candidate(m.group(1))})
    return {h: c for h, c in docs.items() if c >= min_docs}


def update_mapping(vocab, path, write=True):
    if path.exists():
        mapping = pd.read_csv(path, keep_default_na=False)
    else:
        mapping = pd.DataFrame(columns=["header", "group", "n_patents_strict", "source"])
    known = set(mapping["header"])
    new = pd.DataFrame([
        {"header": h, "group": group_by_keywords(h), "n_patents_strict": c, "source": "auto"}
        for h, c in vocab.items() if h not in known
    ])
    if len(new):
        mapping = pd.concat([mapping, new], ignore_index=True)
    mapping = mapping.sort_values(["group", "n_patents_strict"], ascending=[True, False])
    if write:
        mapping.to_csv(path, index=False)
    return dict(zip(mapping["header"], mapping["group"])), len(new)


def split_run(words, mapping):
    out, i = [], 0
    while i < len(words):
        for j in range(len(words), i, -1):
            if " ".join(words[i:j]) in mapping:
                out.append((i, j))
                i = j
                break
        else:
            return None
    return out


def find_headers(text, mapping):
    out = []
    for m in LOOSE_RE.finditer(text):
        tokens = list(re.finditer(r"\S+", m.group(1)))
        while tokens and (re.search(r"\d", tokens[0].group(0)) or LEADING_JUNK_RE.fullmatch(tokens[0].group(0))):
            tokens.pop(0)
        if not tokens:
            continue
        words = [t.group(0) for t in tokens]
        base = m.start(1)

        parts = split_run(words, mapping)
        if parts is not None:
            for i, j in parts:
                header = " ".join(words[i:j])
                out.append({"group": mapping[header], "header": header, "source": "vocab", "start": base + tokens[i].start()})
            continue

        header = " ".join(words)
        group = group_by_keywords(header)
        if group != "other" and sum(c.isalpha() for c in header) >= MIN_LETTERS:
            out.append({"group": group, "header": header, "source": "keywords", "start": base + tokens[0].start()})
    return out
