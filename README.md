# Comparing Chunking Strategies for Patent Documents in Semantic Search and RAG

[![License](https://badgen.net/github/license/provvalek/rag-scipapers?color=green)](https://github.com/provvalek/rag-scipapers/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/provvalek/rag-scipapers)](https://github.com/provvalek/rag-scipapers/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/provvalek/rag-scipapers.svg?color=0088ff)](https://github.com/provvalek/rag-scipapers/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/provvalek/rag-scipapers.svg?color=7f29d6)](https://github.com/provvalek/rag-scipapers/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Elizaveta Kozhemiakova </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Roman Ischenko </td>
    </tr>
</table>

## About

Before a patent can be searched with embeddings or fed to a RAG system, it has to be cut into pieces. Usually this is done by length, and nobody asks which pieces are actually worth embedding. This project checks whether cutting a patent along its own structure (sections and blocks such as "object of the invention" or "technical effect") finds relevant prior art better than cutting it by length or by paragraphs, and which sections matter most for search.

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

The quality of semantic search and retrieval-augmented generation (RAG) over long technical documents largely depends on how a document is split into fragments (chunks) before embedding. This work looks for a way to chunk and embed patent documents that improves retrieval quality with off-the-shelf text embedding models, without fine-tuning them. A patent is a long and heterogeneous document. Its sections (summary of the invention, prior art, detailed description, claims) differ greatly in how informative they are, and fixed-length chunking mixes them together, blurs the meaning of each fragment and lets boilerplate text dominate the search results. We propose to form chunks along the semantic structure of the document itself, by sections and by blocks marked with trigger phrases, rather than by text length. On a corpus of 37,727 patents, these structure-aware schemes are compared with fixed-length chunking, paragraph grouping and hierarchical indexing built on language model summaries. Patent citations of prior-art documents serve as expert relevance labels. We also estimate how much each section contributes to retrieval quality. The results make it possible to determine which parts of a patent document should be embedded first. In practice, the work gives recommendations on choosing a chunking scheme for search, clustering and RAG over scientific and technical information.

**Keywords:** retrieval-augmented generation, dense retrieval, document chunking, patent documents, text embeddings

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{kozhemiakova2026chunking,
    title={Comparing Chunking Strategies for Patent Documents in Semantic Search and RAG},
    author={Kozhemiakova, Elizaveta and Ischenko, Roman},
    year={2026}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
