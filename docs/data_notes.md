# Data sources and quality notes

## Sources
- **Demand:** EIA Open Data API v2, hourly PJM demand, 2023-01-01 to 2025-12-31 (UTC). Reported by the grid operator and subject to revision.
- **Weather:** Open-Meteo historical archive (reanalysis, not station observations). Temperature in C, wind in km/h. Five PJM load centers: Philadelphia, Washington DC, Pittsburgh, Columbus, Chicago. They are averaged into one regional series.

## Quality findings (raw demand)
- 26,303 rows for 26,304 expected hours: 1 hour missing entirely.
- 46 rows present with no value. Together with the missing hour, the gaps fall on 2023-11-05/06 and 2024-03-10/11, the dates of the U.S. clock changes. This points to a time-handling artifact in the source, though the cause has not been confirmed.
- One value (2024-11-21 17:00 UTC, 56,260 MWh) looks low for a weekday afternoon and is checked by the suspect-value rule.
- Weather: complete, no nulls, plausible ranges.

## Cleaning rules (sql/01_demand_hourly.sql)
1. Build a complete hourly grid.
2. Flag a value as suspect if it is more than 30 percent below (or above) both the same hour 24h earlier and the same hour 24h later. Requiring both neighbors to agree stops one bad value from also flagging its good neighbors.
3. Fill missing or suspect hours with the average of the same hour 7 days earlier and later, and flag them (`is_imputed`).
4. Exclude imputed hours from model scoring.
