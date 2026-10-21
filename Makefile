.PHONY: test demo evaluate index-benchmark

test:
	python -m unittest discover -s tests -v

demo:
	python -m hard_negative_retrieval demo --output artifacts/demo.json

evaluate:
	python -m hard_negative_retrieval evaluate --config experiments/configs/default.json --output artifacts/evaluation.json --checkpoint artifacts/model.json

index-benchmark:
	PYTHONPATH=src python -m hard_negative_retrieval index benchmark --artifact-directory artifacts/late-interaction --output artifacts/index-benchmark.json
