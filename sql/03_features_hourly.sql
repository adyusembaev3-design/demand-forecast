-- Step 3c: model-ready feature table.
-- Calendar features use US Eastern local time. Lags use only values known 24h or more
-- before the target hour, so they are available at forecast time.
-- Rows without a full week of history are dropped.
CREATE OR REPLACE TABLE features_hourly AS
WITH base AS (
    SELECT
        d.ts,
        d.ts_local,
        d.demand_mwh,
        d.is_imputed,
        d.is_suspect,
        w.temp_c,
        w.temp_c_max_city,
        w.temp_c_min_city,
        w.humidity_pct,
        w.wind_kmh
    FROM demand_hourly d
    LEFT JOIN weather_hourly w ON w.ts = d.ts
),
feat AS (
    SELECT
        *,
        hour(ts_local) AS hour_local,
        dayofweek(ts_local) AS dow,
        month(ts_local) AS month_num,
        (dayofweek(ts_local) IN (0, 6)) AS is_weekend,
        greatest(temp_c - 18.3, 0) AS cdd,
        greatest(18.3 - temp_c, 0) AS hdd,
        lag(demand_mwh, 24) OVER w AS lag_24,
        lag(demand_mwh, 48) OVER w AS lag_48,
        lag(demand_mwh, 168) OVER w AS lag_168,
        avg(demand_mwh) OVER (
            ORDER BY ts ROWS BETWEEN 47 PRECEDING AND 24 PRECEDING
        ) AS roll24_mean
    FROM base
    WINDOW w AS (ORDER BY ts)
)
SELECT *
FROM feat
WHERE lag_168 IS NOT NULL;
