-- mission_summary.sql
-- Per-rotor performance summary from a sim run.
-- Run against any dash_results_<timestamp>.db
--
--   sqlite3 results_<ts>/dash_results_<ts>.db < mission_summary.sql


WITH rotor_kw AS (
    SELECT timestamp_s, 1 AS rotor_id, kw_r1 AS kw FROM telemetry UNION ALL
    SELECT timestamp_s, 2,             kw_r2        FROM telemetry UNION ALL
    SELECT timestamp_s, 3,             kw_r3        FROM telemetry UNION ALL
    SELECT timestamp_s, 4,             kw_r4        FROM telemetry UNION ALL
    SELECT timestamp_s, 5,             kw_r5        FROM telemetry UNION ALL
    SELECT timestamp_s, 6,             kw_r6        FROM telemetry
)
SELECT
    p.airport__icao               AS dep,
    p.destination__icao           AS arr,
    p._generated__distance_km     AS dist_km,
    p.fixed_wing__dash_speed_kmh  AS cruise_kmh,
    p.fixed_wing__dash_altitude_m AS cruise_alt_agl_m,
    r.rotor_id,
    r.R_m,
    r.P_max_kW                    AS motor_ceiling_kw,
    r.notes                       AS rotor_variant,
    ROUND(SUM(k.kw) * 0.1 / 1000, 3) AS rotor_energy_MJ,
    ROUND(MAX(k.kw), 1)           AS rotor_max_power_kW,
    MAX(t.gz)                     AS max_gz
FROM test_parameters p
CROSS JOIN rotor_config r
JOIN rotor_kw k ON k.rotor_id = r.rotor_id
JOIN telemetry t ON t.timestamp_s = k.timestamp_s
GROUP BY r.rotor_id;