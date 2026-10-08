.PHONY: install test test-nonfed test-nonfed-unit smoke smoke-nonfed smoke-notebooks-nonfed benchmark validate-cross-modal cross-modal-cli clean

# Source-only CI: no downloaded cohort or sealed research-result fixtures.
# Keep historical evidence tests in NONFED_TESTS; do not silently skip them.
NONFED_UNIT_TESTS = \
	tests/test_accelerated.py \
	tests/test_datasets.py \
	tests/test_metrics.py \
	tests/test_vp.py \
	tests/test_randomized.py \
	tests/test_randomized_reduced.py \
	tests/test_randomized_reduced_fast.py \
	tests/test_robust.py \
	tests/test_seed_contract.py \
	tests/test_torch_backend.py \
	tests/test_legacy_torch_contract.py \
	tests/test_genomics_aim_protocol.py \
	tests/test_genomics_blocked.py \
	tests/test_genomics_cached.py \
	tests/test_genomics_benchmark.py \
	tests/test_genomics_snp_budget.py \
	tests/test_video_experiment.py \
	tests/test_video_benchmark.py \
	tests/test_video_paper_reproduction.py \
	tests/test_cleaned_encoder_v2.py \
	tests/test_certificate_balanced_cleaned_encoder.py \
	tests/test_genomics_selected5_launcher.py

NONFED_TESTS = \
	tests/test_accelerated.py \
	tests/test_audit_video_cleaned_input_solver_blind_pack_provenance.py \
	tests/test_certificate_balanced_cleaned_encoder.py \
	tests/test_cleaned_encoder_v2.py \
	tests/test_combine_genomics_benchmark_shards.py \
	tests/test_datasets.py \
	tests/test_download_genomics_all_autosome_sources.py \
	tests/test_evaluation_assets.py \
	tests/test_genomics_all_autosome_formal_results.py \
	tests/test_genomics_amr_whitening_formal_results.py \
	tests/test_genomics_aim_protocol.py \
	tests/test_genomics_aim_protocol_cli.py \
	tests/test_genomics_aim_formal_results.py \
	tests/test_genomics_aim_presentation.py \
	tests/test_genomics_benchmark.py \
	tests/test_genomics_blocked.py \
	tests/test_genomics_cached.py \
	tests/test_genomics_default_runtime_results.py \
	tests/test_genomics_data_provenance.py \
	tests/test_genomics_population_report.py \
	tests/test_genomics_randomized_vp_matched.py \
	tests/test_genomics_randomized_vp_5pct.py \
	tests/test_genomics_randomized_vp_5pct_results.py \
	tests/test_genomics_randomized_vp_selected5.py \
	tests/test_genomics_randomized_vp_selected5_results.py \
	tests/test_genomics_randomized_vp_amr_results.py \
	tests/test_genomics_pca_loading_all_autosomes.py \
	tests/test_genomics_pca_loading_amr_whitening.py \
	tests/test_genomics_pca_loading_stratification_extension.py \
	tests/test_genomics_scale_aware_vp_confirmation.py \
	tests/test_genomics_selected5_launcher.py \
	tests/test_genomics_all_autosome_runtime_benchmark.py \
	tests/test_genomics_pca_loading_grid.py \
	tests/test_genomics_pca_loading_confirmation.py \
	tests/test_genomics_two_stage_solver_feasibility.py \
	tests/test_genomics_snp_budget.py \
	tests/test_genomics_write_contract.py \
	tests/test_legacy_torch_contract.py \
	tests/test_metrics.py \
	tests/test_nonfed_release_artifact.py \
	tests/test_plink_raw_to_npz.py \
	tests/test_prepare_genomics_aim_1kg.py \
	tests/test_prepare_genomics_aim_all_autosomes.py \
	tests/test_prepare_video_cleaned_input_solver_blind_pack.py \
	tests/test_render_genomics_amr_loadings.py \
	tests/test_benchmark_randomized_runtime.py \
	tests/test_randomized_runtime_evidence.py \
	tests/test_randomized.py \
	tests/test_randomized_reduced.py \
	tests/test_randomized_reduced_fast.py \
	tests/test_repository_contract.py \
	tests/test_robust.py \
	tests/test_seed_contract.py \
	tests/test_spca_default_pca_addendum.py \
	tests/test_spca_default_pca_addendum_results.py \
	tests/test_summarize_spca_benchmark.py \
	tests/test_torch_backend.py \
	tests/test_validate_nonfed_release.py \
	tests/test_video_benchmark.py \
	tests/test_video_cleaned_encoder_v2.py \
	tests/test_video_cleaned_input_solver_truth_blind_smoke.py \
	tests/test_video_confirmation_contrast_v2.py \
	tests/test_video_figure5_frozen_noise_stress.py \
	tests/test_video_figure5_historical_github_current.py \
	tests/test_video_figure5_paper_style.py \
	tests/test_video_github_style_current_gallery.py \
	tests/test_video_confirmation_presentation.py \
	tests/test_video_experiment.py \
	tests/test_video_literal_cohort.py \
	tests/test_video_paper_reproduction.py \
	tests/test_video_paper_robust_study.py \
	tests/test_video_visual_receipt.py \
	tests/test_video_visualization.py \
	tests/test_vp.py

NONFED_SMOKE_ROOT = artifacts/smoke/nonfed

install:
	python -m pip install -e '.[experiments,dev]'

test:
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/vp_rspca_mpl pytest

test-nonfed:
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/vp_rspca_mpl pytest $(NONFED_TESTS)

test-nonfed-unit:
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/vp_rspca_mpl pytest $(NONFED_UNIT_TESTS)

smoke:
	python experiments/spca_benchmark.py --repeats 1 --base-seed 2025 --n-samples 80 --n-features 40 --rank 3 --alphas 0.1 --max-iter 50 --certified-full-refine-iters 20
	python experiments/federated_synthetic.py --seeds 2025 --n-clients 8 --input-dim 20 --output-dim 16
	python experiments/federated_20news.py --help
	python experiments/video_spca.py --grid-size 10 --n-timesteps 30 --max-iter 10 --randomized-full-refine-iters 2 --output-dir artifacts/smoke/video_spca

smoke-nonfed:
	python experiments/spca_benchmark.py --repeats 1 --base-seed 2025 --n-samples 80 --n-features 40 --rank 3 --alphas 0.1 --max-iter 50 --certified-full-refine-iters 20 --output-json $(NONFED_SMOKE_ROOT)/spca/results.json --output-csv $(NONFED_SMOKE_ROOT)/spca/metrics.csv
	python experiments/video_spca.py --grid-size 10 --n-timesteps 30 --max-iter 10 --randomized-full-refine-iters 2 --output-dir $(NONFED_SMOKE_ROOT)/video
	python experiments/video_spca_benchmark.py --repeats 2 --base-seed 2025 --methods pca,vp,robust --grid-size 8 --n-timesteps 16 --max-iter 2 --bootstrap-resamples 32 --bootstrap-seed 2025 --output-json $(NONFED_SMOKE_ROOT)/video-paired/results.json --output-csv $(NONFED_SMOKE_ROOT)/video-paired/metrics.csv
	python tools/generate_genomics_smoke_data.py --output-dir $(NONFED_SMOKE_ROOT)/genomics/input
	python experiments/genomics_1kg.py $(NONFED_SMOKE_ROOT)/genomics/input/synthetic_genotypes.npz --orientation samples-by-snps --missing-value -9 --method pca --rank 2 --bootstrap-repeats 1 --metadata $(NONFED_SMOKE_ROOT)/genomics/input/synthetic_populations.csv --output-json $(NONFED_SMOKE_ROOT)/genomics/results.json --output-csv $(NONFED_SMOKE_ROOT)/genomics/metrics.csv

smoke-notebooks-nonfed:
	python tools/execute_nonfed_notebooks.py --output-dir $(NONFED_SMOKE_ROOT)/notebooks

benchmark:
	python experiments/spca_benchmark.py --repeats 5 --base-seed 2025 --output-json artifacts/spca_benchmark/results.json --output-csv artifacts/spca_benchmark/metrics.csv

validate-cross-modal:
	python tools/validate_cross_modal_research_program.py
	pytest -q \
		tests/test_cross_modal_research_program.py \
		tests/test_cross_modal_gates.py \
		tests/test_trust_audit.py \
		tests/test_vision_data.py \
		tests/test_federated_clip.py \
		tests/test_prepare_federated_vision_data.py \
		tests/test_prepare_20news_train_only_data.py \
		tests/test_federated_vision_trust_audit.py \
		tests/test_federated_20news_trust_audit.py \
		tests/test_federated_trust_mechanism_audit.py \
		tests/test_cross_modal_mechanism_factories.py \
		tests/test_cross_modal_mechanism_factory_strict_e2e.py \
		tests/test_federated_gauge_representation_audit.py \
		tests/test_cross_modal_remote_plan.py

cross-modal-cli:
	python -m experiments.prepare_federated_vision_data --help
	python -m experiments.prepare_20news_train_only_data --help
	python -m experiments.federated_vision_trust_audit --help
	python -m experiments.federated_20news_trust_audit --help
	python -m experiments.federated_trust_mechanism_audit --help
	python -m experiments.federated_gauge_representation_audit --help
	python -m operations.cross_modal_remote_plan --help
	python -m operations.run_cross_modal_remote_job --help

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]"
