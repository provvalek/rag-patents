import re

SENT_END = re.compile(r"[.!?]+\s")
BEFORE_FIRST = "before_first_header"


def section_spans(headers, text_len):
    starts = [h["start"] for h in headers]
    spans = [(BEFORE_FIRST, 0, starts[0])] if starts and starts[0] > 0 else []
    spans += [(h["group"], a, e) for h, a, e in zip(headers, starts, starts[1:] + [text_len])]
    return spans


def measure(text):
    return {
        "chars": len(text),
        "words": len(text.split()),
        "sentences": len(SENT_END.findall(text)) + (1 if text.strip() else 0),
    }


def chunk_spans(text, sent_per_chunk, overlap):
    starts = [0] + [m.end() for m in SENT_END.finditer(text) if m.end() < len(text)]
    step = max(1, sent_per_chunk - overlap)
    spans = []
    for i in range(0, len(starts), step):
        end = starts[i + sent_per_chunk] if i + sent_per_chunk < len(starts) else len(text)
        spans.append((starts[i], end))
        if i + sent_per_chunk >= len(starts):
            break
    return spans


def crossing(chunks, boundaries):
    return [any(a < b < e for b in boundaries) for a, e in chunks]
