# Retrieval runs and denominators

There are two different recorded workloads.

| Workload | Corpus and queries | What it measures |
| --- | --- | --- |
| Mean-pooling encoder | 32 documents; 96 train, 32 development, 64 test queries | Ranking with learned alias associations and disjoint query templates |
| Token interaction and IVF | 96 documents; 160 train, 128 test queries | MaxSim ranking, approximate-index agreement, and relation-binding failures |

The small encoder run in [the reference report](../benchmarks/reference/evaluation.json) reaches Recall@1 of 1.0. The random-negative ablation reaches the same result, so it does not establish an advantage for hard mining. Shared products, intents, aliases, and corpus text make this a learnable closed-corpus experiment.

```sh
python -m hard_negative_retrieval evaluate --config experiments/configs/default.json --ablation \
  --output artifacts/negative-strategies.json
python -m hard_negative_retrieval index benchmark --epochs 8 --clusters 8 --queries 128 \
  --output artifacts/index-results.json
```

The [index report](../reports/index-benchmark.json) retains 64 alias and 64 relation-binding queries. Exhaustive MaxSim gets 74/128 top-one relevance hits; full probing agrees with that imperfect oracle exactly. One probe and a 16-candidate limit reduce token-dot work at the cost of agreement. The [late-interaction results](late-interaction-results.md) preserve the complete measured settings, architectural failures, and timing scope.

For another run, record corpus/query fingerprints, parameter identities, seed, hyperparameters, negative strategy, probes, candidates, and all query-level contributions. CPU timings include Python validation and bookkeeping; dot-product counts are work counters, not hardware instructions. A public-data result needs its own licensed input and independent queries.
