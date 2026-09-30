# traffic-intel (backend)

Python package for the AI Urban Traffic & Incident Intelligence system.
See the repository root [README](../README.md), [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
and [docs/TRAFFIC_STATE_ESTIMATION.md](../docs/TRAFFIC_STATE_ESTIMATION.md).

```bash
pip install -e ".[dev,cv]"
python -m traffic_intel.pipeline synthetic --scenario default
python -m traffic_intel.pipeline replay cam_01 --annotate
```
