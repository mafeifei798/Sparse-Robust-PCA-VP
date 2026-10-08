"""Fail-closed cross-modal gates for the preregistered FedRPCA trust audit."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping


DATASETS = ("eurosat", "svhn", "20news")
VISION_DATASETS = frozenset(("eurosat", "svhn"))
SEMANTIC_CONDITIONS = (
    "benign_client_concentrated_concept",
    "harmful_histogram_preserving_label_reassignment",
)
SAVED_ROUNDS = (1, 10, 25, 50)
DESCRIPTIVE_ROUNDS = (1, 10, 25)
PRIMARY_INFERENTIAL_ROUND = 50
PRIMARY_ORIGINAL_BLOCK_COUNT = 48
DEVELOPMENT_SEED_COUNT = 3


def _finite_real(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real non-boolean number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _exact_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a boolean")
    return bool(value)


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(f"{field} keys changed; missing={missing}, extra={extra}")


def _validate_condition_map(
    payload: Any,
    *,
    fields: set[str],
    name: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _exact_keys(payload, set(SEMANTIC_CONDITIONS), field=name)
    validated: dict[str, dict[str, Any]] = {}
    for condition in SEMANTIC_CONDITIONS:
        row = payload[condition]
        if not isinstance(row, Mapping):
            raise TypeError(f"{name}.{condition} must be a mapping")
        _exact_keys(row, fields, field=f"{name}.{condition}")
        validated[condition] = dict(row)
    return validated


def evaluate_semantic_dataset_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and gate one dataset without selecting a layer, block, or round."""

    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    expected = {
        "dataset",
        "protocol_passed",
        "support_auprc_over_prevalence_ratio",
        "support_auprc_delta_vs_best_simple_score",
        "fault_client_auroc",
        "fault_client_auroc_delta_vs_best_simple_score",
        "counterfactual_direction",
        "component_interventions",
        "direction_bootstrap_protocol",
        "component_effect_source",
        "saved_rounds",
        "primary_inferential_round",
        "descriptive_rounds",
        "descriptive_rounds_entered_gate",
        "primary_round_original_block_count",
        "development_seed_count",
        "all_predeclared_blocks_reported",
        "layer_or_round_selection_performed",
    }
    _exact_keys(evidence, expected, field="evidence")
    dataset = evidence["dataset"]
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    protocol = _exact_bool(evidence["protocol_passed"], field="protocol_passed")
    all_blocks = _exact_bool(
        evidence["all_predeclared_blocks_reported"],
        field="all_predeclared_blocks_reported",
    )
    selected = _exact_bool(
        evidence["layer_or_round_selection_performed"],
        field="layer_or_round_selection_performed",
    )
    rounds = evidence["saved_rounds"]
    if isinstance(rounds, (str, bytes)) or tuple(rounds) != SAVED_ROUNDS:
        raise ValueError("saved_rounds must equal (1, 10, 25, 50)")
    if (
        type(evidence["primary_inferential_round"]) is not int
        or evidence["primary_inferential_round"] != PRIMARY_INFERENTIAL_ROUND
    ):
        raise ValueError("primary_inferential_round must equal 50")
    descriptive_rounds = evidence["descriptive_rounds"]
    if isinstance(descriptive_rounds, (str, bytes)) or (
        tuple(descriptive_rounds) != DESCRIPTIVE_ROUNDS
    ):
        raise ValueError("descriptive_rounds must equal (1, 10, 25)")
    descriptive_entered_gate = _exact_bool(
        evidence["descriptive_rounds_entered_gate"],
        field="descriptive_rounds_entered_gate",
    )
    if descriptive_entered_gate:
        raise ValueError("descriptive rounds may not enter the primary semantic gate")
    if (
        type(evidence["primary_round_original_block_count"]) is not int
        or evidence["primary_round_original_block_count"]
        != PRIMARY_ORIGINAL_BLOCK_COUNT
    ):
        raise ValueError("primary round must report exactly 48 original blocks")
    if (
        type(evidence["development_seed_count"]) is not int
        or evidence["development_seed_count"] != DEVELOPMENT_SEED_COUNT
    ):
        raise ValueError("development_seed_count must equal 3")
    if evidence["component_effect_source"] != (
        "original_dataset_seed_condition_round_50_atoms_only"
    ):
        raise ValueError("component effects must use only original round-50 atoms")

    bootstrap = evidence["direction_bootstrap_protocol"]
    if not isinstance(bootstrap, Mapping):
        raise TypeError("direction_bootstrap_protocol must be a mapping")
    _exact_keys(
        bootstrap,
        {
            "replicates",
            "rpca_recomputed_each_draw",
            "validation_evaluator_calls_per_draw",
            "target",
        },
        field="direction_bootstrap_protocol",
    )
    if type(bootstrap["replicates"]) is not int or bootstrap["replicates"] != 1000:
        raise ValueError("direction bootstrap must use exactly 1000 replicates")
    if (
        _exact_bool(
            bootstrap["rpca_recomputed_each_draw"],
            field="direction_bootstrap_protocol.rpca_recomputed_each_draw",
        )
        is not True
    ):
        raise ValueError("direction bootstrap must recompute RPCA for every draw")
    if (
        type(bootstrap["validation_evaluator_calls_per_draw"]) is not int
        or bootstrap["validation_evaluator_calls_per_draw"] != 0
    ):
        raise ValueError("direction bootstrap may not call a validation evaluator")
    if bootstrap["target"] != "counterfactual_direction_alignment_only":
        raise ValueError("bootstrap target must be direction alignment only")

    support_ratio = _finite_real(
        evidence["support_auprc_over_prevalence_ratio"],
        field="support_auprc_over_prevalence_ratio",
    )
    support_delta = _finite_real(
        evidence["support_auprc_delta_vs_best_simple_score"],
        field="support_auprc_delta_vs_best_simple_score",
    )
    fault_auroc = _finite_real(
        evidence["fault_client_auroc"], field="fault_client_auroc"
    )
    fault_delta = _finite_real(
        evidence["fault_client_auroc_delta_vs_best_simple_score"],
        field="fault_client_auroc_delta_vs_best_simple_score",
    )
    if not 0.0 <= fault_auroc <= 1.0:
        raise ValueError("fault_client_auroc must lie in [0, 1]")

    directions = _validate_condition_map(
        evidence["counterfactual_direction"],
        fields={"paired_mean_alignment", "bootstrap_95_ci_lower"},
        name="counterfactual_direction",
    )
    components = _validate_condition_map(
        evidence["component_interventions"],
        fields={
            "full_effect_paired_mean",
            "full_minus_L_only_paired_mean",
            "signed_permutation_empirical_p",
            "random_sparse_empirical_p",
            "full_exceeds_all_signed_permutation_draws",
            "full_exceeds_all_random_sparse_draws",
        },
        name="component_interventions",
    )

    reasons: list[str] = []
    if not protocol:
        reasons.append("protocol_failed")
    if not all_blocks:
        reasons.append("incomplete_block_registry")
    if selected:
        reasons.append("post_hoc_layer_or_round_selection")
    if support_ratio < 2.0:
        reasons.append("support_auprc_ratio_below_2")
    if support_delta < 0.02:
        reasons.append("support_delta_below_0.02")
    if fault_auroc < 0.75:
        reasons.append("fault_auroc_below_0.75")
    if fault_delta < 0.05:
        reasons.append("fault_delta_below_0.05")
    for condition in SEMANTIC_CONDITIONS:
        mean_alignment = _finite_real(
            directions[condition]["paired_mean_alignment"],
            field=f"counterfactual_direction.{condition}.paired_mean_alignment",
        )
        lower = _finite_real(
            directions[condition]["bootstrap_95_ci_lower"],
            field=f"counterfactual_direction.{condition}.bootstrap_95_ci_lower",
        )
        if mean_alignment <= 0.0:
            reasons.append(f"{condition}:direction_mean_not_positive")
        if lower <= 0.0:
            reasons.append(f"{condition}:direction_ci_lower_not_positive")

        component = components[condition]
        full_effect = _finite_real(
            component["full_effect_paired_mean"],
            field=f"component_interventions.{condition}.full_effect_paired_mean",
        )
        full_delta = _finite_real(
            component["full_minus_L_only_paired_mean"],
            field=f"component_interventions.{condition}.full_minus_L_only_paired_mean",
        )
        signed_p = _finite_real(
            component["signed_permutation_empirical_p"],
            field=f"component_interventions.{condition}.signed_permutation_empirical_p",
        )
        random_p = _finite_real(
            component["random_sparse_empirical_p"],
            field=f"component_interventions.{condition}.random_sparse_empirical_p",
        )
        if not 0.0 <= signed_p <= 1.0 or not 0.0 <= random_p <= 1.0:
            raise ValueError("component empirical p-values must lie in [0, 1]")
        signed_exceeds = _exact_bool(
            component["full_exceeds_all_signed_permutation_draws"],
            field=(
                f"component_interventions.{condition}."
                "full_exceeds_all_signed_permutation_draws"
            ),
        )
        random_exceeds = _exact_bool(
            component["full_exceeds_all_random_sparse_draws"],
            field=(
                f"component_interventions.{condition}."
                "full_exceeds_all_random_sparse_draws"
            ),
        )
        if full_effect <= 0.0:
            reasons.append(f"{condition}:full_effect_not_positive")
        if full_delta <= 0.0:
            reasons.append(f"{condition}:full_not_better_than_L_only")
        if signed_p > 0.05 or not signed_exceeds:
            reasons.append(f"{condition}:signed_permutation_null_not_rejected")
        if random_p > 0.05 or not random_exceeds:
            reasons.append(f"{condition}:random_sparse_null_not_rejected")
    return {
        "dataset": dataset,
        "passed": not reasons,
        "reasons": reasons,
        "eligible_for_cross_modal_semantic_gate": True,
    }


def evaluate_cross_modal_semantic_gate(
    evidence_by_dataset: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require semantic PASS on 20News and one vision task, reporting all three."""

    if not isinstance(evidence_by_dataset, Mapping):
        raise TypeError("evidence_by_dataset must be a mapping")
    _exact_keys(evidence_by_dataset, set(DATASETS), field="evidence_by_dataset")
    decisions: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        row = dict(evidence_by_dataset[dataset])
        if row.get("dataset") != dataset:
            raise ValueError(f"dataset identity mismatch for {dataset!r}")
        decisions[dataset] = evaluate_semantic_dataset_evidence(row)
    passing = tuple(dataset for dataset in DATASETS if decisions[dataset]["passed"])
    passed = "20news" in passing and bool(set(passing) & VISION_DATASETS)
    reasons: list[str] = []
    if "20news" not in passing:
        reasons.append("20news_did_not_pass")
    if not (set(passing) & VISION_DATASETS):
        reasons.append("no_vision_dataset_passed")
    return {
        "passed": passed,
        "passing_datasets": list(passing),
        "required_modalities": "20news_and_at_least_one_vision",
        "dataset_decisions": decisions,
        "reasons": reasons,
        "safe_amplification_stage_authorized": passed,
        "cv_performance_claim_authorized": False,
    }


__all__ = [
    "DATASETS",
    "DESCRIPTIVE_ROUNDS",
    "DEVELOPMENT_SEED_COUNT",
    "PRIMARY_INFERENTIAL_ROUND",
    "PRIMARY_ORIGINAL_BLOCK_COUNT",
    "SAVED_ROUNDS",
    "SEMANTIC_CONDITIONS",
    "evaluate_cross_modal_semantic_gate",
    "evaluate_semantic_dataset_evidence",
]
