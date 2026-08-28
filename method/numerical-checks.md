# Numerical and artifact regressions

## Reproduced defects and fixes

1. **Duplicate metric cutoffs inflated scores above one.** `evaluate_retriever(..., cutoffs=(1,1))` returned Recall@1, MRR@1, and nDCG@1 of 2.0. Cutoffs are now validated as nonempty and unique before retrieval; each query contributes exactly once per named metric.
2. **Finite inputs could yield NaN InfoNCE outputs.** Query `[1e308]` and document vectors `[1e308]`, `[-1e308]` overflowed their dot products, bypassed the finite-input check, and returned NaN loss/gradients. Scores, final loss, and gradients now fail explicitly if nonfinite. Numeric guards also reject booleans, wrong types, and integers too large for finite floating conversion.
3. **Scalar relevance strings could be interpreted as document-ID characters.** An imported `"relevant":"a"` could become `('a',)` and be accepted for a single-character document ID. Import now requires arrays, while direct dataset/training APIs require typed tuple labels without duplicates.
4. **Malformed checkpoint traces and matrix containers were insufficiently validated.** Loading now validates version type, list-of-list weight containers, bounded finite numeric values, and the training fingerprint trace before exposing a model.
5. **Zero-information queries could enter training.** A stopword-only query produced no token features and no useful query updates. Training now rejects such input explicitly.
6. **Evaluation trusted nonfinite retriever scores.** Ranking adapters returning NaN/Infinity now fail validation rather than allowing an apparently valid aggregate report.

## Added capabilities

- A controlled `--ablation` experiment trains hard- and uniform-random-negative models with the same corpus, training labels, hyperparameters, initialization, and query-update order. Negative sampling uses a separate seeded RNG and excludes all positives.
- `--load-checkpoint` evaluates without retraining and requires both corpus and training-query/label fingerprints to match the configured dataset.
- Reports include per-query ranked document IDs and metric contributions. Aggregate scores are independently reconstructible, rather than only a final average.
- Epoch-order hashes make update-order matching visible in ablation reports. Configuration reads are bounded to 64 KiB.

The gradient derivation, mean-pooling normalization, positive exclusion, and split boundaries were independently inspected. Existing finite-difference tests verify both encoder gradients. Training sees the shared corpus and training queries only; query-template holdouts remain explicit and are not relabeled as unseen-domain evaluation.

## Executed verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m hard_negative_retrieval evaluate --config experiments/configs/default.json --ablation --output benchmarks/reference/evaluation.json --checkpoint benchmarks/reference/model.json
PYTHONPATH=src python -m hard_negative_retrieval evaluate --load-checkpoint benchmarks/reference/model.json --output artifacts/reloaded.json
ruff check --select E9,F src tests
ruff format --check src tests
```

The reviewed suite has **37 passing tests**, including 13 new regressions covering metric inflation, overflow, strict numeric/label types, zero-signal training, random negative safety, identical ablation order, malformed checkpoints, raw metric reconstruction, CLI ablation, and checkpoint split mismatch.

The executed 32-document synthetic benchmark has 96 training, 32 development, and 64 test queries. Both hard-negative and random-negative training reach Recall@1 of 1.0 on its 64 test queries. This is evidence that the implementation runs on this easy controlled corpus, **not evidence of an advantage for hard mining or real-world language generalization**.

Three small faulty variants were tested in isolated temporary source copies: allowing duplicate cutoffs, accepting boolean numeric settings, and accepting scalar relevance strings. Behavioral tests detected all three; [spot-check evidence](mutation-spotcheck.json) lists the failing tests. Original source files were not mutated.

TestProof was not installed; `npx --no-install testproof --format json --fail-under 100` failed before running it, and nothing was downloaded. The manual spot checks are a small selected sample, not a full mutation score or exhaustive proof.

## Remaining limitations

This is a small exact-search, mean-pooling model, not a pretrained semantic encoder or ANN service. Documents and aliases are shared across query-template splits; the dataset is intentionally easy. A checkpoint hash detects accidental mismatch but does not authenticate an adversarial artifact. The optional SentenceTransformer adapter remains separate, lazy, and unmeasured. Stronger claims require independent data, multiple seeds, and controlled compute measurements.
