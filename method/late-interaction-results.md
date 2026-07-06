# Late interaction and index results

## New core capabilities

- Real token-level cosine MaxSim using immutable copies of the existing trained query/document token tables.
- Exhaustive exact scoring with deterministic tie-breaking and exact token-interaction work counters.
- Seeded spherical k-means++ over document coarse vectors, learned centroids, and corpus-complete document postings.
- Explicit cluster probes and candidate budgets, followed by exact MaxSim reranking.
- Index persistence bound to corpus and full source-model/scoring fingerprints, with centroid/partition/assignment validation.
- Executable build/search/benchmark CLI paths and a per-repository CI probe-consistency smoke run.

The upgrade reuses actually trained non-contextual token embeddings. It does not pretend to train a contextual/ColBERT model, and it makes no MaxSim-specific optimization claim. See [maxsim-and-partitions.md](maxsim-and-partitions.md) for scoring equations, index algorithms, budgets, and complexity.

## Executed nontrivial experiment

```bash
PYTHONPATH=src python -m hard_negative_retrieval index benchmark --epochs 8 --clusters 8 --queries 128 --artifact-directory artifacts/late-interaction --output reports/index-benchmark.json
PYTHONPATH=src python -m hard_negative_retrieval index search --dataset artifacts/late-interaction/dataset.json --checkpoint artifacts/late-interaction/model.json --index artifacts/late-interaction/index.json --query 'Within project0 find the rule allowing blue teams to export tables' --probes 2 --max-candidates 16 --output reports/index-search.json
PYTHONPATH=src python -m unittest discover -s tests -v
```

The benchmark trains on **160 queries over 96 documents**, then evaluates **128 held-out queries** with eight coarse clusters. Half the queries require relation binding between colors and permissions; the corresponding document pairs have identical token multisets with roles swapped. This exposes an architectural failure instead of preserving the earlier easy all-perfect dataset.

Exhaustive MaxSim ground-truth relevance@1 is **74/128**: **64/64** alias cases and **10/64** relation-binding cases. Agreement with exhaustive search is measured separately:

| Probes | Maximum candidates | Oracle recall@5 | Mean candidates reranked | Mean exact token dots |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0.74375 | 10.8125 | 482.875 |
| 2 | 96 | 0.921875 | 45.28125 | 2066.3125 |
| 8 | 96 | 1.0 | 96.0 | 3836.0 |

The full-probe result is an implementation consistency check, not a perfect-accuracy claim. The small-budget path saves interaction work but can miss useful documents. Full probing adds coarse overhead and can be slower than exhaustive scoring. [The actual report](../reports/index-benchmark.json) contains every query's rankings, relevance labels, work, one-run CPU timings, model/data hashes, centroid bucket sizes, and a complete probe/candidate sweep.

## Verification

The full suite has **48 passing tests**, including eleven new index/scoring tests. Independent scalar examples verify MaxSim with positive, negative, and repeated query contributions. A hand-constructed case places the best multi-token document far from the coarse query mean: one probe misses it, full probing recovers it, and a tight candidate cap can still lose it. Other tests cover OOV handling, strict budgets, seeded clustering, frozen encoder state, stale checkpoint/index rejection, corrupted postings, full-probe equality, and explicit role-binding failure.

Existing finite-difference tests still check both trained embedding tables' InfoNCE gradients. This upgrade does not claim a new MaxSim training gradient. Reports distinguish exact token dot products and coarse dot products; these are not FLOPs, database reads, or latency guarantees. Local timings include Python validation/bookkeeping and are not a production load test.

## Remaining limits

The exact/index paths are CPU Python baselines with at most 1,000 documents and bounded token/dimension/work budgets. They do not implement token compression, incremental index maintenance, distributed ANN serving, contextual encoding, or supervised late-interaction fine-tuning. Hashes detect mismatched artifacts but do not authenticate an adversarial publisher. More independent domains, contextual models, multiple seeds, and controlled load measurements are needed for broader accuracy/performance claims.
