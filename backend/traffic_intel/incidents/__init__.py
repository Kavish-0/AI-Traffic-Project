"""Incident intelligence (Phase 6): anomaly evidence -> hedged incident hypotheses -> explainable risk.

    engine.py     IntelligenceEngine - per closed window: series detectors, fusion, forecast, risk
    scenarios.py  six synthetic scenarios run through the real pipeline (tests, demo)
    __main__.py   CLI: scenarios | ingest <run dir> | what-if

Decision support only: every incident carries a probability and an uncertainty note, never a verdict.
"""
