import collections
import random

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .data import iter_patents
from .headers import find_headers, group_by_keywords


def collect(path, vocab, n_contexts, context_chars, seed, limit=0):
    rng = random.Random(seed)
    mapping = {h: h for h in vocab}
    positions = collections.defaultdict(list)
    contexts = collections.defaultdict(list)
    seen = collections.Counter()
    for _, text in iter_patents(path, limit):
        hits = [h for h in find_headers(text, mapping) if h["source"] == "vocab"]
        for h, nxt in zip(hits, hits[1:] + [None]):
            header, start = h["header"], h["start"]
            positions[header].append(start / max(len(text), 1))
            context = text[start + len(header):(nxt["start"] if nxt else len(text))][:context_chars].strip()
            if not context:
                continue
            seen[header] += 1
            if len(contexts[header]) < n_contexts:
                contexts[header].append(context)
            else:
                j = rng.randrange(seen[header])
                if j < n_contexts:
                    contexts[header][j] = context
    return positions, contexts


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def featurize(headers, positions, contexts, model, cfg):
    name_emb = np.array(model.encode([h.lower() for h in headers], normalize_embeddings=True))
    flat = [(i, c) for i, h in enumerate(headers) for c in contexts.get(h, [])]
    ctx_emb = np.zeros_like(name_emb)
    if flat:
        vecs = np.array(model.encode([c for _, c in flat], batch_size=128, normalize_embeddings=True))
        idx = np.array([i for i, _ in flat])
        for i in range(len(headers)):
            if (idx == i).any():
                ctx_emb[i] = vecs[idx == i].mean(axis=0)
    ctx_emb = l2(ctx_emb)
    pos = np.array([np.quantile(positions[h], [0.1, 0.5, 0.9]) for h in headers])
    pos_block = pos / np.sqrt(pos.shape[1])
    X = np.hstack([cfg["w_name"] * name_emb, cfg["w_context"] * ctx_emb, cfg["w_position"] * pos_block])
    return X, pos


def cluster_agglomerative(X, rule_groups, cfg):
    scan = []
    for k in range(cfg["k_range"][0], cfg["k_range"][1] + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        scan.append({"k": k, "silhouette": round(silhouette_score(X, labels), 4),
                     "ari_vs_rules": round(adjusted_rand_score(rule_groups, labels), 4)})
    labels = AgglomerativeClustering(n_clusters=cfg["n_clusters"], linkage="ward").fit_predict(X)
    return labels, pd.DataFrame(scan)


def cluster_dbscan(X, rule_groups, cfg, seed):
    import umap

    Z = umap.UMAP(n_components=cfg["umap_dim"], n_neighbors=cfg["umap_neighbors"], min_dist=0.0,
                  random_state=seed).fit_transform(X)
    knn_dist = NearestNeighbors(n_neighbors=cfg["min_samples"] + 1).fit(Z).kneighbors(Z)[0][:, -1]
    scan = []
    for q in np.linspace(0.05, 0.95, 19):
        eps = float(np.quantile(knn_dist, q))
        labels = DBSCAN(eps=eps, min_samples=cfg["min_samples"]).fit_predict(Z)
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        sil = silhouette_score(Z[mask], labels[mask]) if n_clusters >= 2 else float("nan")
        scan.append({"eps": round(eps, 4), "knn_quantile": round(q, 2), "n_clusters": n_clusters,
                     "noise_share": round(1 - mask.mean(), 4), "silhouette_non_noise": round(sil, 4),
                     "ari_vs_rules": round(adjusted_rand_score(rule_groups, labels), 4)})
    scan = pd.DataFrame(scan)
    eps = cfg["eps"]
    if eps <= 0:
        ok = scan[(scan["noise_share"] <= cfg["max_noise"]) & (scan["n_clusters"] >= cfg["min_clusters"])
                  & scan["silhouette_non_noise"].notna()]
        if ok.empty:
            raise ValueError("нет подходящего eps: см. eps_scan.csv и задайте CLUSTERS['eps'] в config.py")
        eps = float(ok.loc[ok["silhouette_non_noise"].idxmax(), "eps"])
    return DBSCAN(eps=eps, min_samples=cfg["min_samples"]).fit_predict(Z), scan


def run_clusters(path, vocab, cfg, out, seed, limit=0):
    from sentence_transformers import SentenceTransformer

    out.mkdir(parents=True, exist_ok=True)
    positions, contexts = collect(path, vocab, cfg["n_contexts"], cfg["context_chars"], seed, limit)
    headers = sorted(h for h in vocab if positions.get(h))
    X, pos = featurize(headers, positions, contexts, SentenceTransformer(cfg["model"]), cfg)
    rule_groups = [group_by_keywords(h) for h in headers]

    if cfg["method"] == "dbscan":
        import umap

        labels, scan = cluster_dbscan(X, rule_groups, cfg, seed)
        scan.to_csv(out / "eps_scan.csv", index=False)
        xy = umap.UMAP(n_components=2, n_neighbors=cfg["umap_neighbors"], min_dist=0.1,
                       random_state=seed).fit_transform(X)
    else:
        labels, scan = cluster_agglomerative(X, rule_groups, cfg)
        scan.to_csv(out / "k_scan.csv", index=False)
        xy = PCA(n_components=2, random_state=seed).fit_transform(X)

    df = pd.DataFrame({
        "header": headers,
        "cluster": labels,
        "rule_group": rule_groups,
        "n_patents_strict": [vocab[h] for h in headers],
        "n_occurrences": [len(positions[h]) for h in headers],
        "pos_q10": pos[:, 0].round(4), "pos_q50": pos[:, 1].round(4), "pos_q90": pos[:, 2].round(4),
        "example_context": [(contexts.get(h) or [""])[0][:200] for h in headers],
        "x": xy[:, 0].round(4), "y": xy[:, 1].round(4),
    }).sort_values(["cluster", "n_patents_strict"], ascending=[True, False])
    df.to_csv(out / "header_clusters.csv", index=False)

    summary = []
    for c, g in df.groupby("cluster"):
        majority = g.groupby("rule_group")["n_patents_strict"].sum().sort_values(ascending=False)
        summary.append({
            "cluster": c,
            "size": len(g),
            "n_patents_strict": int(g["n_patents_strict"].sum()),
            "pos_median": round(float(np.average(g["pos_q50"], weights=g["n_patents_strict"])), 4),
            "majority_rule_group": majority.index[0],
            "majority_share": round(float(majority.iloc[0] / majority.sum()), 4),
            "top_headers": " | ".join(g["header"].head(8)),
        })
    summary = pd.DataFrame(summary).sort_values("pos_median")
    summary.to_csv(out / "clusters_summary.csv", index=False)
    return summary
