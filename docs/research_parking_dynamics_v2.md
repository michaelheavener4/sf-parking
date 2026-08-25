# Research frame: blockface parking dynamics

The system is split into three layers: transaction-to-occupancy state estimation, short-horizon occupancy forecasting, and driver decision support.

Meter transactions are a noisy proxy for physical occupancy because payment and actual dwell can differ. Transaction-based occupancy research therefore models payment behavior and finds that reliable estimation depends on spatial granularity.

The forecasting literature supports finite-capacity queueing models, non-homogeneous arrival rates, time-varying service-time distributions, and regime changes. The current benchmark intentionally stays white-box before returning to machine learning.

V2 label: `active paid sessions / mapped blockface capacity`.
V2 predictors: active sessions at T-1, training-window blockface/hour-of-week arrival intensity, empirical conditional session-duration survival, mapped capacity.
Champion baseline: exact one-hour persistence on the same target.

The transaction-implied target is not claimed to be physical ground truth. Production calibration should use independent occupancy observations when available.
