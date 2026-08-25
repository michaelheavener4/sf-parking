# ML Runbook

## 1. Install ML dependencies

```bash
cd ~/sf-parking
python -m pip install -e '.[ml]'
```

If your checkout reports `does not provide the extra 'ml'`, run `git pull --ff-only` first. The current `pyproject.toml` defines the ML extra.

## 2. Run the complete spatial/dynamic tournament

```bash
PYTHONPATH="$PWD/src:$PWD/scripts" python scripts/train_spatial_paid_state.py \
  --train-days 90 \
  --validation-days 7 \
  --test-days 7 \
  --max-train-rows 250000 \
  --max-validation-rows 100000 \
  --max-test-rows 150000 \
  --neighbor-k 24 \
  --neighbor-radius-m 250
```

The tournament compares persistence, hour-conditioned climatology, the existing 15-feature LightGBM, and the 31-feature spatial/dynamic LightGBM. It reports overall MAE/RMSE plus availability bands, transition periods, daytime, and peak-evening performance.

The spatial candidate is automatically promoted only when it beats both the current LightGBM and persistence overall **and** beats both on the transition regime with at least 100 transition observations. Otherwise the candidate remains isolated and production is unchanged. The process exits `3` when the promotion gate fails; that is an intentional scientific result, not a crash.

## 3. Generate spatial/dynamic T+1 forecasts

Only run this after a candidate has been promoted:

```bash
PYTHONPATH="$PWD/src" python scripts/forecast_spatial_paid_state.py
```

The spatial model is deliberately T+1 only. The established recursive model remains the safe T+2–T+24 path until recursive spatial features are explicitly validated.

## 4. Probability calibration

The tournament creates a validation-only spatial calibrator. After forecasts mature, the production calibration workflow can be run against matured forecasts:

```bash
PYTHONPATH="$PWD/src" python scripts/fit_parking_probability_calibrator.py \
  --threshold 0.50
```

Calibration estimates `P(actual availability >= 50%)`. Raw regression output is never described as a probability without calibration.

## 5. Decision engine

For the spatial model, use its matching calibrator explicitly:

```bash
PYTHONPATH="$PWD/src" python scripts/find_parking_intelligent.py \
  --lat 37.7972638 --lon -122.4334589 \
  --date 2026-08-24 --hour 18 \
  --radius 250 --top 10 \
  --model-version spatial_dynamic_v1 \
  --calibrator models/paid_state_spatial_probability_calibrator.json
```

The finder ranks opportunities using forecast probability and distance, and reports a correlated neighborhood success estimate. It never silently jumps to a different forecast slot.

## 6. Production health

```bash
PYTHONPATH="$PWD/src" python scripts/check_ml_health.py --hours 48
```

## Promotion philosophy

Never optimize aggregate MAE alone. The model is useful when it predicts transitions and high-demand states. A model that improves the 30–90% availability regime and peak-demand hours while preserving calibration is more valuable than one that merely reduces error on an overwhelmingly 100%-available test set.
