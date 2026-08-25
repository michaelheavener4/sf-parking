# Blockface transaction dynamics V2 research frame

The benchmark separates transaction-derived state estimation from forecasting and driver decision support.

Meter transactions are a noisy proxy for physical occupancy because payment behavior and physical dwell can differ. Transaction-based occupancy work therefore models payment behavior and shows a spatial-granularity tradeoff.

Queueing-based parking forecasting uses finite-capacity systems, non-homogeneous arrivals, time-varying service times, and regime changes. V2 uses blockface-scale transaction-implied occupancy as the target and keeps exact one-hour persistence as the benchmark champion.

Target: active paid sessions / mapped blockface parking capacity.
Predictors available at T-1: active paid sessions, training-only arrival intensity by blockface and hour-of-week, empirical conditional duration survival, and mapped capacity.
