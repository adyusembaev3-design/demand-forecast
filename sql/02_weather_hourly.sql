-- Step 3b: collapse the five city series into one regional hourly weather series.
-- Input : raw_weather (ts, city, temperature_2m, relative_humidity_2m, wind_speed_10m)
-- Output: weather_hourly
CREATE OR REPLACE TABLE weather_hourly AS
SELECT
    ts,
    avg(temperature_2m) AS temp_c,
    max(temperature_2m) AS temp_c_max_city,
    min(temperature_2m) AS temp_c_min_city,
    avg(relative_humidity_2m) AS humidity_pct,
    avg(wind_speed_10m) AS wind_kmh,
    count(*) AS n_cities
FROM raw_weather
GROUP BY ts;
