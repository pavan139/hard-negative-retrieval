# Sharing an encoder and index

An index file alone is insufficient. Supply the corpus, training/query split, model checkpoint, index settings, and the command that produced the report. The corpus and encoder fingerprints must agree with the saved index. Loading a mismatched artifact should fail; do not bypass the check for a release.

```sh
python -m pip install -e . -r requirements-dev.txt
python bin/check_retriever.py --require-lint --wheel
python -m hard_negative_retrieval index benchmark --epochs 8 --clusters 8 --queries 128 \
  --artifact-directory artifacts/index --output artifacts/index-results.json
```

Keep per-query relevance and approximate-search agreement in separate fields. The [recorded index report](../reports/index-benchmark.json) is an example of the distinction. A different model, tokenization rule, corpus order, or score definition needs a new bound artifact and fresh measurements.

The CI workflow runs the relocated verifier and a small index benchmark. Local check results live in `reports/verification.json`; a remote Actions run is separate evidence. Review licenses before distributing imported text or pretrained weights. Publish only the small reference artifacts needed for the documented experiment, leaving environments, build directories, and unrelated checkpoints out of the source tree.
