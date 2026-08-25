# Parking dynamics research frame

## Research conclusion

The project should separate three layers:

1. **State estimation** from transactions. Meter transactions are not a perfect occupancy sensor; payment behavior can involve underpayment, overpayment, unpaid parking, and other deviations. Yang & Qian (2017) explicitly model payment behavior to turn transactions into time-varying occupancy and report a spatial-granularity tradeoff.
2. **Occupancy forecasting** from an explicit stochastic process. Prior parking research uses finite-capacity queueing models, time-varying arrival/departure behavior, and regime changes. Xiao, Lou & Frisby (2018) use an M/M/C/C framework in San Francisco; later work extends this to time-varying arrival/departure regimes. Tavafoghi, Poolla & Varaiya (2019) use non-homogeneous arrivals and time-varying service times.
3. **Decision support** for a driver: forecast availability at the desired arrival horizon and spatial radius, rather than optimize an arbitrary one-hour probability error.

## Current benchmark target

The V2 blockface benchmark uses a transaction-implied occupancy label:

`active paid sessions on blockface / mapped parking-space capacity`

The one-hour forecast uses only information available at T-1:

- active paid sessions at T-1
- training-window arrival intensity by blockface and hour-of-week
- training-window empirical session-duration survival
- mapped blockface capacity

The benchmark keeps exact one-hour persistence as the champion baseline.

## Important limitation

The transaction-implied target is still not the same thing as physical occupancy. A paid session may not equal a vehicle's exact physical dwell interval. Therefore, a future production system should eventually calibrate the transaction-derived latent occupancy estimate against independent sensor/manual occupancy observations where available.
