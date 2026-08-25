# Blockface transaction dynamics V2

Run:

```bash
python3 -m pytest -q tests/test_blockface_transaction_dynamics_v2.py
python3 scripts/benchmark_blockface_transaction_dynamics_v2.py
```

The benchmark compares exact one-hour persistence against a white-box blockface forecast built from active paid sessions at T-1, empirical conditional session survival, time-varying arrival intensity, and mapped blockface capacity.

The target is explicitly **transaction-implied occupancy**, not claimed physical occupancy. The current research literature supports transaction-based occupancy inference but documents payment-behavior and spatial-granularity limitations.
