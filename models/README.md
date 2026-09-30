# models/

Versioned model artifacts (not committed). One directory per trained model:

    models/<task>/<name>/<version>/
        model.pt | model.json | model.joblib   # weights
        metadata.json                          # ModelMetadata: params, metrics, feature_schema_hash, git sha
    models/<task>/<name>/PRODUCTION            # text file containing the promoted version

`detection/` also holds pretrained detector weights (e.g. `yolo11s.pt`).
Online code only ever loads the PRODUCTION version; promotion is done with
`python -m traffic_intel.ai.training promote ...` after evaluation.
