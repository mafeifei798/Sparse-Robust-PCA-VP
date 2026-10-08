# Contributing

Changes to an algorithm must include:

1. the objective and update rule in the docstring;
2. a deterministic unit test on a small matrix;
3. a regression test for objective/stationarity diagnostics;
4. a paired comparison against the closest baseline when performance is claimed;
5. an explicit note when a heuristic does not inherit a published guarantee.

Do not copy implementations into notebooks.  Notebooks and experiment scripts
must import the package under `src/rspca` so every result exercises tested code.

Generated figures, downloaded datasets, feature caches, and model checkpoints
belong under ignored artifact directories.  Commit tidy metrics and manifests
only when they are intentionally part of a reproducible result release.

