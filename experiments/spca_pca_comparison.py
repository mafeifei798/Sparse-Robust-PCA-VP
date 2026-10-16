"""Paired comparison of default PCA and sparse PCA implementations."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
import warnings
import zipfile

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["pca_default", "pca_uncentered", "thresholded_pca_matched_sparsity",
           "vp_spca", "accelerated_vp_spca", "randomized_vp_spca", "certified_randomized_vp_spca"]
SPARSE = METHODS[3:]
SOURCE_BINDINGS = {
    "vp": (None,
           "cae985e15e87d5ebf88c9c68c7570ed10832df30ec1245b7b8031def2bd6b9c8"),
    "accelerated": ("src/rspca/accelerated.py", "594b17a826c0416f4f1b09b0c3228369a73e88a5c1ed80184ddfb09b7c5afcb7"),
    "randomized": (None,
                   "37b68488d25abf775f97ec7ed755a1f5b00aceaa6fd08e29d044fe01e39fe0c5"),
    "datasets": ("src/rspca/datasets.py", "4b72f51579616bf91b8bbdb4929d2caa41a330c3b104d42de5fa4e501532e154"),
    "metrics": ("src/rspca/metrics.py", "35f3be948cac70d34c1e180a7684f1e5c4c30d1b291fbd956d010d784c864dda"),
}
HISTORICAL_RESULT_SHA256 = {
    "clean": "7a829415888ade2c0b2ca9655e6650ccbcd3ac77632382277f1fa26f784865e2",
    "corrupt": "1af95c9a055c96012cb753e56361a65bed10242647e041db9ab63f4c21a0a7f1",
}
DATA_KEYS = ["n_samples", "n_features", "rank", "support_fraction", "noise_std",
             "corruption_fraction", "corruption_scale", "corruption_distribution"]
METRICS = ["subspace_error", "relative_reconstruction_error", "achieved_sparsity",
           "support_precision", "support_recall", "support_f1"]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def strict_json(path):
    def reject(value):
        raise ValueError(f"Nonstandard JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=convert)+"\n")


def bind_sources(vp_source, randomized_source):
    """Resolve supplied archives while retaining every historical byte hash."""
    supplied = {"vp": Path(vp_source), "randomized": Path(randomized_source)}
    return {name: (supplied[name] if relative is None else ROOT / relative, expected)
            for name, (relative, expected) in SOURCE_BINDINGS.items()}


@contextmanager
def historical_modules(bindings):
    """Import exactly the five bound historical modules, without current __init__."""
    prefix = "_ordinary_pca_historical_rspca"
    if any(name == prefix or name.startswith(prefix+".") for name in sys.modules):
        raise RuntimeError("Historical package already loaded")
    for source, expected in bindings.values():
        if digest(source) != expected:
            raise ValueError(f"Historical source mismatch: {source}")
    package = ModuleType(prefix)
    package.__path__ = []
    sys.modules[prefix] = package
    loaded = {}
    try:
        for short, (source, _) in bindings.items():
            name = prefix+"."+short
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            # SourceFileLoader writes __pycache__ beside sealed historical
            # sources. Compile authenticated bytes directly so this import
            # cannot mutate an execution snapshot or its closed file set.
            source_bytes = source.read_bytes()
            exec(compile(source_bytes, str(source), "exec", dont_inherit=True), module.__dict__)
            loaded[short] = module
        yield SimpleNamespace(**loaded)
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix+"."):
                del sys.modules[name]


def load_history(history_root):
    history_root = Path(history_root)
    history = {}
    for regime, sha in HISTORICAL_RESULT_SHA256.items():
        path = history_root/f"spca/{regime}/results.json"
        if digest(path) != sha:
            raise ValueError(f"Historical record mismatch: {regime}")
        value = strict_json(path)
        c = value["config"]
        if (c["n_samples"], c["n_features"], c["rank"], c["repeats"], c["max_iter"], c["tol"]) != (400, 200, 5, 12, 500, 1e-5):
            raise ValueError("Original configuration changed")
        seeds = value["seed_manifest"]["seeds"]
        if len(seeds) != 12 or len(set(seeds)) != 12:
            raise ValueError("Original seed bank is incomplete")
        index = {(r["seed"], r["method"]): r for r in value["records"]}
        for seed in seeds:
            for method in METHODS[2:]:
                if (seed, method) not in index:
                    raise ValueError("Missing original locked method")
        history[regime] = value
    return history


def threshold_loadings(B, target):
    flat = np.abs(B).reshape(-1)
    keep = int(round((1-target)*flat.size))
    result = np.zeros_like(B)
    if keep == 0:
        return result
    if keep >= flat.size:
        return B.copy()
    selected = np.argpartition(flat, flat.size-keep)[-keep:]
    result.reshape(-1)[selected] = B.reshape(-1)[selected]
    return result


def fit_method(method, X, config, alpha, seed, legacy):
    rank = config["rank"]
    started = perf_counter()
    if method == "pca_default":
        model = PCA(n_components=rank).fit(X)
        B = model.components_.T.copy()
        A, mean = B.copy(), model.mean_.copy()
        diagnostic = dict(parameters=model.get_params(), resolved_solver=model._fit_svd_solver,
                          converged=True, status="closed_form_default_pca", n_iter=1)
    elif method in METHODS[1:3]:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        B = vt[:rank].T.copy()
        A = B.copy()
        if method == METHODS[2]:
            B = threshold_loadings(B, config["target_sparsity"])
            left, _, right = np.linalg.svd(X.T @ (X @ B), full_matrices=False)
            A = left @ right
        mean = np.zeros(X.shape[1])
        diagnostic = dict(converged=True, status="closed_form_supplementary_baseline", n_iter=1)
    else:
        common = dict(alpha=alpha, max_iter=config["max_iter"], tol=config["tol"], verbose=False)
        if method == "vp_spca":
            model = legacy.vp.fit_vp_spca(X, rank, **common)
        elif method == "accelerated_vp_spca":
            model = legacy.accelerated.fit_accelerated_vp_spca(X, rank, **common,
                acceleration=True, backtracking=True, monotone=True, adaptive_restart=True)
        else:
            common.update(oversample=config["oversample"], n_power_iter=config["n_power_iter"], random_state=seed)
            if method == "randomized_vp_spca":
                model = legacy.randomized.fit_randomized_vp_spca(X, rank, **common,
                    use_full_gradient=config["randomized_full_gradient"],
                    finalize_full_iters=config["randomized_finalize_full_iters"])
            elif method == "certified_randomized_vp_spca":
                model = legacy.randomized.certified_randomized_vp_spca(X, rank, **common,
                    residual_tolerance=config["certified_residual_tolerance"],
                    full_stationarity_tolerance=config["tol"],
                    full_refine_iters=config["certified_full_refine_iters"])
            else:
                raise ValueError(method)
        A, B = model.A, model.B
        mean = np.zeros(X.shape[1])
        stationarity = float(model.full_stationarity if hasattr(model, "full_stationarity") else model.stationarity[-1])
        objective = float(model.full_objective if hasattr(model, "full_objective") else model.objectives[-1])
        diagnostic = dict(converged=bool(model.converged), n_iter=int(model.n_iter), status=model.status,
            final_stationarity=stationarity, final_objective=objective,
            objective_trace=list(model.objectives), stationarity_trace=list(model.stationarity))
        for attr in ("sketch_converged", "full_converged", "full_refinement_iterations", "sketch_residual_relative"):
            if hasattr(model, attr):
                diagnostic[attr] = getattr(model, attr)
        if getattr(model, "certificate", None) is not None:
            diagnostic["certificate"] = asdict(model.certificate)
    fit_seconds = perf_counter()-started
    scores = (X-mean) @ B
    reconstruction = scores @ A.T + mean
    if not all(np.isfinite(a).all() for a in (A, B, mean, scores, reconstruction)):
        raise ValueError("Nonfinite model output")
    return dict(A=A, B=B, mean=mean, scores=scores, reconstruction=reconstruction,
                fit_seconds=fit_seconds, diagnostic=diagnostic)


def evaluate(data, fit, legacy):
    metrics = legacy.metrics
    match = metrics.component_correlation(data.loadings, fit["B"])
    aligned = np.zeros_like(data.loadings)
    for ref, estimate, sign in zip(match["reference_indices"], match["estimated_indices"], match["signs"], strict=True):
        aligned[:, ref] = sign*fit["B"][:, estimate]
    support = metrics.support_precision_recall_f1(data.feature_support, aligned)
    return dict(subspace_error=metrics.subspace_error(data.loadings, fit["B"]),
        relative_reconstruction_error=metrics.relative_reconstruction_error(data.low_rank, fit["reconstruction"]),
        achieved_sparsity=metrics.sparsity_ratio(fit["B"]),
        support_precision=support["precision"], support_recall=support["recall"], support_f1=support["f1"])


def reproduction_check(record, original):
    thresholded = record["method"] == "thresholded_pca_matched_sparsity"
    tolerance = 1e-8 if thresholded else 1e-4
    errors = {k: abs(record[k]-original[k]) for k in METRICS[:3]}
    passed = (errors["subspace_error"] <= tolerance and errors["relative_reconstruction_error"] <= tolerance
              and errors["achieved_sparsity"] <= .002 and record["alpha"] == original["alpha"])
    return dict(regime=record["regime"], seed=record["seed"], method=record["method"],
        status="passed" if passed else "failed", error_tolerance=tolerance,
        sparsity_tolerance=.002, absolute_differences=errors,
        historical_converged=original["converged"], local_converged=record["diagnostic"]["converged"])


def summarize(records, resamples=100000):
    cells, contrasts = [], []
    for regime in ("clean", "corrupt"):
        seeds = list(dict.fromkeys(r["seed"] for r in records if r["regime"] == regime))
        index = {(r["seed"], r["method"]): r for r in records if r["regime"] == regime}
        draw = np.random.default_rng(260826).integers(0, len(seeds), size=(resamples, len(seeds)))
        for method in METHODS:
            rows = [index[s, method] for s in seeds]
            cells.append(dict(regime=regime, method=method, n_seeds=len(seeds),
                converged_count=sum(r["diagnostic"]["converged"] for r in rows),
                means={k: float(np.mean([r[k] for r in rows])) for k in METRICS},
                sample_std={k: float(np.std([r[k] for r in rows], ddof=1)) for k in METRICS},
                fit_seconds_median=float(np.median([r["fit_seconds"] for r in rows]))))
            for baseline in METHODS[:3]:
                if method == baseline:
                    continue
                other = [index[s, baseline] for s in seeds]
                qualified = all(r["diagnostic"]["converged"] for r in rows+other)
                result = dict(regime=regime, method=method, baseline=baseline, seeds=seeds,
                              inference_scope="exposed_seed_descriptive_not_confirmation",
                              all_pairs_converged=qualified, metrics={})
                for key in ("subspace_error", "relative_reconstruction_error", "support_f1"):
                    method_values = np.array([r[key] for r in rows])
                    baseline_values = np.array([r[key] for r in other])
                    delta = method_values-baseline_values
                    samples = delta[draw].mean(axis=1)
                    values = dict(method_mean=float(method_values.mean()), baseline_mean=float(baseline_values.mean()),
                        paired_mean_difference=float(delta.mean()),
                        paired_difference_ci95=np.quantile(samples, [.025, .975]).tolist(),
                        method_better_pairs=int(np.sum(delta > 0 if key == "support_f1" else delta < 0)),
                        paired_differences=delta.tolist())
                    if key != "support_f1":
                        ratio = 100*(1-method_values[draw].mean(axis=1)/baseline_values[draw].mean(axis=1))
                        values.update(reduction_percent=float(100*(1-method_values.mean()/baseline_values.mean())),
                                      reduction_percent_ci95=np.quantile(ratio, [.025, .975]).tolist())
                    result["metrics"][key] = values
                contrasts.append(result)
    return dict(cells=cells, contrasts=contrasts, bootstrap_resamples=resamples, bootstrap_seed=260826,
                percentage_definition="100*(1-mean(method)/mean(baseline))", runtime_in_bootstrap=False)


def render_report(output, summary, reproduction, identities):
    lines = ["# Ordinary PCA versus the original VP-SPCA implementations", "",
        "This paired comparison includes all 12 registered seeds in both regimes. "
        "Every comparison uses locally regenerated, saved inputs and newly refitted methods. "
        "Historical remote VP metrics are never used as the numerator of a new PCA contrast.", "",
        "Main control: `PCA(n_components=5)`, all other sklearn 1.7.0 defaults unchanged. "
        "No thresholded loading, forced slow solver, or precomputed PCA initialization is given "
        "to the sparse methods. The uncentered SVD and original thresholded control are supplementary.", "",
        "All solver/generator/metric source bytes match the historical release. Historical "
        "input bytes need not match across environments; same source/configuration/seeds are "
        "not described as byte-identical Linux-data reconstruction. All local method inputs "
        "are identical within each paired seed. This is exposed-seed descriptive reproduction, "
        "not a new confirmatory experiment or novel algorithm.", "",
        f"Historical-arm reproduction gate: {sum(r['status']=='passed' for r in reproduction)}/{len(reproduction)} passed. "
        f"Historical observed-matrix byte matches: {sum(r['observed_matches_historical'] for r in identities)}/{len(identities)}.", "",
        "## Complete means", "",
        "Sparsity is the fraction of zero coefficients, not a SNP-compression claim. "
        "PCA's closed-form control status is not a sparse stationarity certificate.", "",
        "| Regime | Method | Subspace error | Clean reconstruction error | Zero coefficients | Support F1 | Converged |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in summary["cells"]:
        m = c["means"]
        lines.append(f"| {c['regime']} | {c['method']} | {m['subspace_error']:.8f} | "
            f"{m['relative_reconstruction_error']:.8f} | {100*m['achieved_sparsity']:.3f}% | "
            f"{m['support_f1']:.6f} | {c['converged_count']}/{c['n_seeds']} |")
    lines += ["", "## All subspace-error contrasts", "",
        "Positive reduction means lower error. Intervals are 100,000 paired-seed percentile "
        "bootstrap intervals without multiplicity correction. Nonconverged contrasts remain "
        "fixed-budget descriptive results. No winner is selected.", "",
        "| Regime | Method | Baseline | Error reduction | Paired 95% reduction interval | All pairs converged |",
        "| --- | --- | --- | ---: | --- | --- |"]
    for r in summary["contrasts"]:
        v = r["metrics"]["subspace_error"]
        lo, hi = v["reduction_percent_ci95"]
        lines.append(f"| {r['regime']} | {r['method']} | {r['baseline']} | {v['reduction_percent']:.3f}% | "
            f"[{lo:.3f}%, {hi:.3f}%] | {r['all_pairs_converged']} |")
    lines += ["", "## Interpretation", "",
        "Ordinary PCA is dense, whereas VP's alpha was already locked by the earlier "
        "sparsity-only selection. Thus the ordinary-PCA comparison is a dense-versus-sparse "
        "recovery comparison, not matched sparsity, and does not establish that acceleration "
        "alone improves statistical recovery. The thresholded control addresses the separate "
        "matched-sparsity question. The former 13.7% belongs only to that former control.", "",
        "Default PCA centers internally and reconstruction includes its learned mean. "
        "VP retains the old no-post-corruption-centering policy; the uncentered rank-5 SVD "
        "companion shows the effect without default centering. All methods share the same "
        "observed input before their own fitting policy.", "",
        "The certified randomized arm includes a full-data accelerated refinement when "
        "its certificate requires it. It is not a sketch-only speed result. This supplement "
        "does not bootstrap runtime, compare local timing with AutoDL, revise video/genomics "
        "results, or claim universal superiority.", "",
        "The full JSON contains every reconstruction/support contrast, convergence "
        "diagnostic, alpha lock, input/source identity and per-seed value. NPZ files retain "
        "all observed/truth arrays and all model loadings/native scores. No source package "
        "has been overwritten.", ""]
    (output/"README.md").write_text("\n".join(lines))


def run(output, protocol, *, history_root, vp_source, randomized_source):
    if sklearn.__version__ != "1.7.0":
        raise ValueError("This declared supplement requires sklearn 1.7.0")
    history_root = Path(history_root)
    bindings = bind_sources(vp_source, randomized_source)
    history = load_history(history_root)
    with historical_modules(bindings) as legacy:
        output.mkdir(parents=True, exist_ok=False)
        records, reproduction, identities = [], [], []
        try:
            spec = dict(status="prefit_exposed_seed_baseline_completion", methods=METHODS,
                source_sha256={name: sha for name, (_, sha) in bindings.items()},
                source_paths={name: f"historical_rspca/{name}.py" for name in bindings},
                runner_sha256=digest(Path(__file__)), protocol_sha256=digest(protocol),
                historical_result_sha256=HISTORICAL_RESULT_SHA256,
                regimes={regime: dict(config=h["config"], seeds=h["seed_manifest"]["seeds"],
                    alpha_locks=[dict(seed=r["seed"], method=r["method"], alpha=r["alpha"])
                                 for r in h["records"] if r["method"] in SPARSE]) for regime, h in history.items()},
                environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                 sklearn=sklearn.__version__, platform=platform.platform()),
                historical_byte_identity_claim=False, primary_baseline="PCA(n_components=5); all other defaults unchanged")
            write_json(output/"specification_before_run.json", spec)
            with zipfile.ZipFile(output/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for name, (source, _) in bindings.items():
                    archive.write(source, f"historical_rspca/{name}.py")
                archive.writestr("historical_rspca/__init__.py", "# Isolated historical module namespace.\n")
                archive.write(Path(__file__), "spca_pca_comparison.py")
                archive.write(protocol, "PROTOCOL.md")
                for regime in history:
                    archive.write(history_root/f"spca/{regime}/results.json", f"history/{regime}.json")
            warnings.simplefilter("error", RuntimeWarning)
            with threadpool_limits(1):
                pools = threadpool_info()
                if any(p["num_threads"] != 1 for p in pools):
                    raise RuntimeError("Uncontrolled thread pool")
                write_json(output/"threadpools.json", pools)
                for regime, h in history.items():
                    c = h["config"]
                    old = {(r["seed"], r["method"]): r for r in h["records"]}
                    for index, seed in enumerate(h["seed_manifest"]["seeds"]):
                        directory = output/regime/f"seed_{seed}"
                        directory.mkdir(parents=True)
                        data = legacy.datasets.generate_sparse_low_rank(**{k: c[k] for k in DATA_KEYS}, seed=seed)
                        arrays = {k: getattr(data, k) for k in ("observed", "uncorrupted", "low_rank", "gaussian_noise",
                                  "corruption", "scores", "loadings", "feature_support", "corruption_support", "singular_values")}
                        np.savez_compressed(directory/"input.npz", **arrays)
                        original = old[seed, METHODS[2]]
                        identity = dict(regime=regime, seed=seed,
                            arrays_sha256={k: legacy.datasets.array_sha256(v) for k, v in arrays.items()},
                            historical_observed_sha256=original["observed_sha256"],
                            observed_matches_historical=legacy.datasets.array_sha256(data.observed)==original["observed_sha256"])
                        identities.append(identity)
                        write_json(directory/"input_identity.json", identity)
                        for method in METHODS:
                            alpha = old[seed, method]["alpha"] if method in SPARSE else None
                            fit = fit_method(method, data.observed, c, alpha, seed, legacy)
                            metrics = evaluate(data, fit, legacy)
                            row = dict(regime=regime, seed=seed, method=method, alpha=alpha,
                                input_sha256=identity["arrays_sha256"]["observed"], **metrics,
                                fit_seconds=fit["fit_seconds"], diagnostic=fit["diagnostic"])
                            records.append(row)
                            np.savez_compressed(directory/f"{method}.npz", **{k: fit[k] for k in ("A", "B", "mean", "scores")})
                            if method in METHODS[2:]:
                                reproduction.append(reproduction_check(row, old[seed, method]))
                        if legacy.datasets.array_sha256(data.observed) != identity["arrays_sha256"]["observed"]:
                            raise ValueError("A fit mutated the common input")
                        write_json(output/"records.json", records)
                        write_json(output/"reproduction.json", reproduction)
                        write_json(output/"input_identities.json", identities)
                        print(f"Completed {regime} paired seed {index+1}/12 ({seed}); all seven methods retained", flush=True)
                summary = summarize(records)
                write_json(output/"summary.json", summary)
                render_report(output, summary, reproduction, identities)
                failed = [r for r in reproduction if r["status"] != "passed"]
                if failed:
                    raise RuntimeError(f"Historical metric reproduction failed in {len(failed)} arms; no performance claim permitted")
            write_json(output/"completion.json", dict(status="complete_ordinary_pca_addendum_not_confirmation",
                records=len(records), datasets=len(identities), reproduction_checks=len(reproduction),
                historical_byte_matches=sum(i["observed_matches_historical"] for i in identities),
                files={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}))
            print(f"Complete: {output}", flush=True)
        except BaseException as error:
            write_json(output/"FAILED.json", dict(status="failed_preserved_do_not_use_for_performance_claims",
                error=repr(error), records_completed=len(records), reproduction_checks_completed=len(reproduction)))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--history", dest="history_root", type=Path, required=True,
                        help="Original result archive containing spca/clean and spca/corrupt")
    parser.add_argument("--vp-source", type=Path, required=True, help="Historical VP-SPCA source file")
    parser.add_argument("--randomized-source", type=Path, required=True, help="Historical randomized VP-SPCA source file")
    args = parser.parse_args(argv)
    return run(args.output, args.protocol, history_root=args.history_root,
               vp_source=args.vp_source, randomized_source=args.randomized_source)


if __name__ == "__main__":
    main()
