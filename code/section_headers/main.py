import argparse
import time
from pathlib import Path

import config
from src.data import iter_patents
from src.headers import build_vocab, find_headers, update_mapping
from src.stats import Collector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.INPUT))
    parser.add_argument("--output", default=str(config.RESULTS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clusters", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    vocab = build_vocab(args.input, config.MIN_DOCS, args.limit)
    mapping, n_new = update_mapping(vocab, config.MAPPING, write=not args.limit)
    print(f"словарь заголовков: {len(vocab)}, таблица групп: {len(mapping)} (новых {n_new})")

    collector = Collector(out, config.SENTENCES_PER_CHUNK, config.CHUNK_OVERLAP)
    for i, (patent_id, text) in enumerate(iter_patents(args.input, args.limit), 1):
        collector.add(patent_id, text, find_headers(text, mapping))
        if i % 5000 == 0:
            print(f"патентов обработано: {i}")
    summary, sections = collector.save(len(vocab), len(mapping))

    print(f"патентов: {summary['n_patents']}, с разделами: {summary['share_with_sections']:.1%}")
    print(f"фрагментов: {summary['n_chunks']}, пересекают границу раздела: "
          f"{summary['share_chunks_crossing_boundary']:.1%}")
    print(sections[["group", "share_patents", "mean_share", "sentences_median"]].to_string(index=False))

    if args.clusters:
        from src.clusters import run_clusters

        print(run_clusters(args.input, vocab, config.CLUSTERS, out / "clusters", config.SEED, args.limit).to_string(index=False))

    print(f"результаты: {out}, время: {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
