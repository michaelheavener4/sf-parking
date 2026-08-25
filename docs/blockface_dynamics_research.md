# Blockface transaction dynamics research frame

Meter transactions are a noisy proxy for physical occupancy; transaction-based occupancy work models payment behavior and documents a spatial-granularity tradeoff. Queueing-based parking forecasts use finite-capacity systems, non-homogeneous arrivals, time-varying service times, and regime changes.

This benchmark therefore uses blockface-scale transaction-implied occupancy and keeps exact persistence as the champion baseline.

Target: active paid sessions / mapped blockface parking capacity.
Predictors at T-1: active paid sessions, training-only arrival intensity by blockface/hour-of-week, empirical conditional duration survival, mapped capacity.
