-- Step 3a: complete hourly demand series with suspect-value flags and gap filling.
-- Inputs : time_dim (complete hourly grid), raw_demand (ts, demand_mwh)
-- Output : demand_hourly
-- Rules  : 1) a value is "suspect" if it is more than 30 percent below (or above) BOTH the same
--             hour 24h earlier and the same hour 24h later. Suspect values are removed
--          2) every missing or removed hour is filled with the average of the same hour
--             7 days earlier and 7 days later (falls back to whichever side exists)
--          3) filled hours are flagged in is_imputed so they can be excluded from scoring
CREATE OR REPLACE TABLE demand_hourly AS
WITH joined AS (
    SELECT
        t.ts,
        t.ts_local,
        CASE WHEN isnan(d.demand_mwh) THEN NULL ELSE d.demand_mwh END AS demand_raw
    FROM time_dim t
    LEFT JOIN raw_demand d ON d.ts = t.ts
),
refs AS (
    SELECT
        *,
        lag(demand_raw, 24) OVER w AS prev_day,
        lead(demand_raw, 24) OVER w AS next_day
    FROM joined
    WINDOW w AS (ORDER BY ts)
),
flagged AS (
    SELECT
        *,
        (demand_raw IS NOT NULL
         AND prev_day IS NOT NULL
         AND next_day IS NOT NULL
         AND ((demand_raw < 0.7 * prev_day AND demand_raw < 0.7 * next_day)
              OR (demand_raw > 1.3 * prev_day AND demand_raw > 1.3 * next_day))) AS is_suspect
    FROM refs
),
cleaned AS (
    SELECT
        ts,
        ts_local,
        demand_raw,
        is_suspect,
        CASE WHEN is_suspect THEN NULL ELSE demand_raw END AS demand_valid
    FROM flagged
),
weekly AS (
    SELECT
        *,
        lag(demand_valid, 168) OVER w AS prev_week,
        lead(demand_valid, 168) OVER w AS next_week
    FROM cleaned
    WINDOW w AS (ORDER BY ts)
)
SELECT
    ts,
    ts_local,
    demand_raw,
    is_suspect,
    coalesce(
        demand_valid,
        (coalesce(prev_week, next_week) + coalesce(next_week, prev_week)) / 2.0
    ) AS demand_mwh,
    (demand_valid IS NULL) AS is_imputed
FROM weekly;
