# ML Runbook

## 1. Install ML dependencies

```bash
cd ~/sf-parking
python -m pip install -e '.[ml]'
```

## 2. Run the complete spatial/dynamic tournament

```bash
PYTHONPATH="$PWD/src" python scripts/train_spatial_paid_state.py \
  --train-days 90 \
  --validation-days 7 \
  --test-days 7 \
  --max-train-rows 250000 \
  --max-validation-rows 100000 \
  --max-test-rows 150000 \
  --neighbor-k 24 \
  --neighbor-radius-m 250
```

The script compares the candidate against persistence and hour climatology and prints performance on the difficult subset (`actual availability < 90%`) plus each local hour.

## 3. Promote only after the gates pass

```bash
PYTHONPATH="$PWD/src" python scripts/train_spatial_paid_state.py --promote
```

Promotion requires aggregate performance not to regress materially and positive improvement over persistence on a sufficiently large difficult subset.

## 4. Generate a spatial/dynamic T+1 forecast

```bash
PYTHONPATH="$PWD/src" python scripts/forecast_spatial_paid_state.py
```

This is deliberately T+1 only. The established recursive model remains the safe T+2–T+24 path until recursive spatial features are explicitly validated.

## 5. Fit probability calibration after forecasts mature

```bash
PYTHONPATH="$PWD/src" python scripts/fit_parking_probability_calibrator.py \
  --threshold 0.50
```

Calibration is an empirical estimate of `P(actual availability >= 50%)`; it is not allowed to silently reuse the training labels.

## 6. Use the decision engine

```bash
PYTHONPATH="$PWD/src" python scripts/find_parking_intelligent.py \
  --lat 37.7972638 --lon -122.4334589 \
  --date 2026-08-24 --hour 18 \
  --radius 250 --top 10
```

The finder reports the best individual opportunities and a correlated neighborhood success estimate. It never silently jumps to a later forecast slot.

## 7. Monitor production health

```bash
PYTHONPATH="$PWD/src" python scripts/check_ml_health.py --hours 48
```

## Promotion philosophy

Never optimize aggregate MAE alone. The model is useful when it predicts transitions. A model that improves the 30–90% availability regime and peak-demand hours while preserving calibration is more valuable than one that reduces error on an overwhelmingly 100%-available test set.
