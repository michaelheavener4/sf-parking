# SF Parking Intelligence Roadmap

## Product thesis

The product is not a parking map. It is a **parking decision engine**:

> Given a destination, arrival time, stay duration, budget and preferences, identify the best legal parking option and estimate the probability of successfully parking there.

The initial city is San Francisco because SFMTA/DataSF exposes unusually rich inventory, regulation, transaction, enforcement, closure and historical parking data.

## Core layers

### 1. Ground truth / legal state

Authoritative or first-party sources determine what parking is legally and operationally possible:

- meter inventory
- meter policies
- street sweeping
- parking regulations
- RPP context
- temporary closures
- off-street facilities

### 2. Observed behavior

These sources describe what people and enforcement actually do:

- meter transactions
- parking citations
- historical SFpark occupancy
- eventual user parking outcomes

### 3. Live state

These sources describe what is happening now:

- garage availability
- traffic incidents
- closures/work zones
- events
- weather
- future partner/private inventory

### 4. Inference

Use the above to estimate:

- blockface demand
- occupancy probability
- vacancy probability
- expected search time
- expected arrival delay
- enforcement/legality confidence
- price and total trip cost

### 5. Decision engine

Rank options using a user-specific objective function:

```text
score = availability_probability
        - travel_cost
        - parking_cost
        - walking_cost
        - enforcement_risk
        - uncertainty_penalty
```

The exact scoring model will evolve; the system must expose its evidence and confidence rather than pretending predictions are facts.

## Data-platform milestones

### M0 — completed foundation

- [x] SFMTA meter ingestion
- [x] meter policy ingestion
- [x] normalized records
- [x] PostgreSQL/PostGIS
- [x] idempotent bulk loading
- [x] nearby-meter spatial query
- [x] policy normalization, including `24:00`
- [x] database tests

### M1 — source registry + provenance

- [x] source registry
- [ ] generic ingestion interface
- [ ] ingestion run metadata
- [ ] raw payload retention strategy
- [ ] source freshness checks
- [ ] source health/status endpoint

### M2 — highest-value behavioral feeds

Implement in this order:

1. meter transactions
2. parking citations
3. street sweeping
4. temporary closures/events
5. managed off-street facilities
6. garage availability

For each adapter:

- fetch
- validate schema
- normalize
- preserve source identifiers
- record retrieval time
- record source timestamp when available
- upsert idempotently
- test representative current rows
- expose coverage/freshness metrics

### M3 — enrichment

- [ ] RPP
- [ ] parking regulations
- [ ] OSM parking topology
- [ ] parking sign/photo evidence
- [ ] 511 traffic
- [ ] Caltrans PeMS
- [ ] weather

### M4 — historical modeling dataset

Create feature tables at meter/blockface/facility/time-window granularity:

- transaction arrival rate
- transaction departure rate
- dwell time distribution
- turnover
- historical occupancy where available
- citation rate
- price
- legal availability
- event intensity
- traffic conditions
- weather

Do not call transaction-derived demand an occupancy measurement. It is a proxy.

### M5 — live parking intelligence

Target API:

```text
GET /parking/search

inputs:
  destination
  arrival_time
  stay_duration
  budget
  walking_limit
  preferences

outputs:
  ranked parking options
  estimated vacancy probability
  expected cost
  legal restrictions
  evidence timestamps
  confidence
```

### M6 — prediction

Start simple:

1. historical block/hour baseline
2. statistical demand model
3. gradient-boosted model with live features
4. temporal/spatial model only if evaluation demonstrates a real gain

Every prediction needs measurable offline evaluation and calibration. A model that sounds intelligent but is poorly calibrated is not acceptable.

### M7 — marketplace

Supply types:

- commercial garages
- municipal/public facilities
- office/building inventory
- private driveways
- private garages
- reserved spaces

The marketplace should be additive to the public-data engine, not required for the core recommendation product.

Potential monetization:

- booking commission
- garage/operator referral revenue
- private-space marketplace fees
- premium consumer features
- B2B parking intelligence/API
- eventually operator demand/pricing analytics

## Product flywheel

```text
public data
    ↓
better parking decisions
    ↓
users
    ↓
search + outcome data
    ↓
better predictions
    ↓
more users
    ↓
parking suppliers
    ↓
more bookable inventory
    ↓
more successful outcomes
    ↓
better data
```

## Non-negotiable engineering rules

1. **Never represent an estimate as live occupancy.** Label inferred state explicitly.
2. **Keep source provenance.** Every important observation must be traceable to its source and retrieval time.
3. **Prefer authoritative legal data.** Supplemental sources can corroborate but should not silently override it.
4. **Preserve raw identifiers.** `POST_ID`, `PARKING_SPACE_ID`, blockface IDs and facility IDs are joins, not disposable fields.
5. **Design for idempotency.** Re-running an adapter must not duplicate observations.
6. **Make freshness visible.** A stale source is different from a healthy source with no records.
7. **Test against current source shapes.** DataSF schemas change; fixtures should include edge cases such as `24:00`.
8. **Build city-neutral interfaces.** San Francisco-specific adapters belong below a generic parking intelligence model.
9. **Measure before adding complexity.** Every prediction feature or model should demonstrate incremental value.
10. **Treat marketplace data separately.** User-generated and transactional data has different privacy, trust, and contractual requirements.
