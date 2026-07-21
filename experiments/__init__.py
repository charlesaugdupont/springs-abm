"""SPRINGS-ABM experiment suite.

Structure:
    experiments/orchestrator.py - shared sweep-execution engine (config overrides,
                                  replicate parallelization, tidy Parquet output)
    experiments/metrics.py      - reusable per-run scalar metrics + post-hoc
                                  complex-systems indicators (dispersion,
                                  early-warning signals)
    experiments/<name>/         - one subpackage per experiment (vaccination,
                                  shocks, care_seeking, ...), each a thin
                                  declaration of what to sweep + which metrics
                                  to record, built on top of orchestrator.py
"""