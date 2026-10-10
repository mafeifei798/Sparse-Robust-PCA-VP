.PHONY: install test test-historical cli-check

install:
	python -m pip install -e '.[experiments,dev]'

# Tiny source-level tests; these do not rerun the saved benchmark studies.
test:
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/vp_rspca_mpl python -m pytest -q -m 'not historical_environment'

# Strict byte-level replay requires the original numerical environment.
test-historical:
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/vp_rspca_mpl python -m pytest -q -m historical_environment

# Parse CLI entry points without fitting a model or launching a benchmark.
cli-check:
	python -m experiments.spca_benchmark --help
	python -m experiments.video_clean_mode_recovery --help
	python -m experiments.video_robust_decomposition --help
	python -m tools.run_genomics_selected5 --help
