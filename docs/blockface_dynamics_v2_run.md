# Blockface dynamics V2 run

```bash
python3 -m pytest -q tests/test_blockface_transaction_dynamics_v2.py
python3 scripts/benchmark_blockface_transaction_dynamics_v2.py
```

The target is transaction-implied occupancy: active paid sessions divided by mapped blockface parking capacity. It is deliberately not described as physical occupancy ground truth.
