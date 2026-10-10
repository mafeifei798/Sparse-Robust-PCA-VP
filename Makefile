.PHONY: install test cli-check

install:
	python -m pip install -e '.[experiments,dev]'

# Tiny source-level tests; these do not rerun the saved benchmark studies.
test:
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/vp_rspca_mpl python -m pytest -q

# Parse CLI entry points without fitting a model or launching a benchmark.
cli-check:
	python experiments/spca_benchmark.py --help
	python experiments/video_figure5_historical_github_current.py --help
	python experiments/video_spca_paper_robust_study.py --help
	python tools/run_genomics_selected5.py --help
