# Blockface transaction dynamics V2

Run:

```bash
python3 -m pytest -q tests/test_blockface_transaction_dynamics_v2.py
python3 scripts/benchmark_blockface_transaction_dynamics_v2.py
```

The benchmark compares:

- exact one-hour persistence of transaction-implied availability
- a white-box blockface dynamics forecast using active paid sessions at T-1, empirical conditional session survival, time-varying blockface arrival rates, and blockface capacity

The test label is reconstructed from raw sessions at the target time; it does **not** reuse `paid_availability_probability` as the label.
