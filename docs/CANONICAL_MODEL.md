# Canonical Spatial/Temporal Model

## Why the curb/blockface is the foundation, not the meter

A meter is a **device**: it can be installed, moved, replaced, or removed
while the place it serves — a stretch of curb where vehicles may stop —
persists. Modelling meters as the geographic root makes history unreadable:
when a meter moves 20 metres, the old model loses the fact that *the same
curb space* was regulated before, during, and after. The canonical hierarchy
therefore anchors identity in physical geography:

```text
street
  -> blockface          (one side of one street between intersections)
      -> curb segment   (a stretch of curb within a blockface)
          -> parking space (one parkable spot)
              -> meter     (a device currently or formerly at a place)
```

Not every curb segment has a parking space; not every parking space has a
meter. Every level exists independently and every link is optional unless the
source itself asserts it.

## Entity table reference

| Table | Meaning | Geometry | Populated from |
|---|---|---|---|
| `streets` | named street | none | inventory `street_id` + `street_name` |
| `blockfaces` | one side of a street between intersections | none yet | inventory `blockface_id` (+ same-row `street_id`, `street_seg_ctrln_id`) |
| `curb_segments` | stretch of curb inside a blockface | optional LineString | **unpopulated** — no SFMTA dataset resolves sub-blockface curbs |
| `parking_spaces` | individual parkable space | optional Point | policies `parkingspaceid` |
| `meters` | payment-device identity | none | inventory `post_id` |
| `meter_placements` | where a meter stood, over time | Point (from inventory lat/lon) | inventory rows |
| `parking_space_meters` | which meter served which space, over time | none | policy rows asserting both ids in one row |

## Source field mapping (verified Aug 2026)

| Source field (dataset) | Canonical target | Notes |
|---|---|---|
| `post_id` (8vzz-qzz9) | `meters.post_id`, `meter_placements.source_post_id` | stable across datasets and years (99.3% of 2017-era ids still current) |
| `blockface_id` (8vzz-qzz9) | `blockfaces.source_blockface_id` | "Blockface (side of street) ID" |
| `street_id` / `street_name` (8vzz-qzz9) | `streets.source_street_id` / `name`; `blockfaces.street_id` | blockface→street is a direct same-row observation |
| `street_seg_ctrln_id` (8vzz-qzz9) | `blockfaces.street_centerline_source_id` | city centerline reference, preserved not joined |
| `latitude`/`longitude`/`shape` (8vzz-qzz9) | `meter_placements.location` | meter position at observation time |
| `data_as_of` (8vzz-qzz9) | `meter_placements.valid_from` | America/Los_Angeles floating timestamp; `-infinity` when unknown |
| `active_meter_flag` (8vzz-qzz9) | `meter_placements.active` | M/T/Y = active per source semantics |
| `ParkingSpaceID`/`parkingspaceid` (qq7v-hds4) | `parking_spaces.source_space_id` | described by SFMTA as the inventory primary key; **absent from all rows of the current inventory snapshot** |
| `postid` (qq7v-hds4) | join to `meters` for `parking_space_meters` | authoritative because one source row asserts both ids |
| `startdate`/`enddate` (qq7v-hds4) | `parking_space_meters.valid_period` | aggregated min/max over asserting rows |

Fields that look like identifiers but are deliberately **not** mapped as
such: `objectid` (DataSF portal row id, changes on reload), `osp_id`
(off-street facilities — out of scope), `ms_pay_station_id`/`ms_space_num`
(multi-space pay-station internals, retained in raw data only).

## Temporal design

* `valid_from`/`valid_until` (+ generated range columns) with GiST exclusion
  constraints prevent overlapping states per entity.
* A newer inventory observation **closes** the previous open-ended placement
  instead of updating it: history is append-only.
  "What was true here at time T?" is a containment query
  (`valid_period @> T`) index-backed by `idx_meter_placements_valid_period`.
* Unknown observation times become `-infinity`, never a fabricated date.
* Projections are recorded in `ingestion_runs` like any other source, with
  per-row `run_id`/`retrieved_at`.

## Identity & unresolved references

Canonical surrogate keys (`*_id bigint identity`) are internal. Source ids
are preserved verbatim (`source_*_id`). Historical transaction post_ids that
match no canonical meter remain visible in
`v_unresolved_transaction_posts` — they are honest observations of retired
or future meters, never deleted or re-mapped by geography.

## Known limitations (documented, not hidden)

1. `curb_segments` has no population source yet.
2. `parking_spaces` carry no geometry (no source provides coordinates).
3. `blockfaces.street_id` uses MAX() if a blockface were observed with
   several streets; current data shows no such case.
4. `parking_space_meters` keeps first-seen validity per (space, meter);
   later policy refreshes with different dates do not rewrite it.
