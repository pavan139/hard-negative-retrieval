# Changing retrieval behavior

Start with the query that changes rank. Add it to the appropriate development or regression split and state its relevant document IDs. Keep the final test split separate from tuning.

Changes to gradients need a numerical check against finite differences. Changes to mining must keep every known positive out of the negative pool and preserve the random-versus-hard update-order comparison. Index changes need full-probe equality with exhaustive MaxSim, partition validation, and checkpoint-mismatch tests.

Use `python -m unittest discover -s tests -v` for the behavior suite. Run Ruff on `src` and `tests`. The complete build check is `python bin/check_retriever.py --require-lint --wheel`; it also exercises an installed wheel outside the checkout.

For measurements, name the dataset fingerprint, encoder identity, seed, probe count, candidate limit, and query denominator. Keep relevance metrics separate from agreement with exhaustive ranking. Publish difficult and failed slices along with aggregates. The existing role-swap cases should not disappear to improve a headline score.

Small reference weights belong with the benchmark that produced them. Put larger external checkpoints outside the repository and document their provenance. [Sharing an index](method/sharing-an-index.md) lists the files needed to reproduce a ranking.
