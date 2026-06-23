-- mission_summary.sql
-- Per-rotor-variant performance summary from a sim run.
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
),

rotor_kw_dt AS (
    SELECT
        rotor_id,
        timestamp_s,
        kw,
        timestamp_s - LAG(timestamp_s) OVER (
            PARTITION BY rotor_id ORDER BY timestamp_s
        ) AS dt_s
    FROM rotor_kw
),

-- Group by (R_m, P_max_kW) as the variant key — notes is NULL in this DB
fleet_peak AS (
    SELECT
        r.R_m,
        r.P_max_kW,
        k.timestamp_s,
        SUM(k.kw)           AS fleet_kw_at_t
    FROM rotor_config r
    JOIN rotor_kw k ON k.rotor_id = r.rotor_id
    GROUP BY r.R_m, r.P_max_kW, k.timestamp_s
),

fleet_peak_max AS (
    SELECT
        R_m,
        P_max_kW,
        MAX(fleet_kw_at_t)  AS rotor_fleet_max_power_kW
    FROM fleet_peak
    GROUP BY R_m, P_max_kW
)

SELECT
    p.airport__icao                                         AS dep,
    p.destination__icao                                     AS arr,
    p._generated__distance_km                               AS dist_km,
    ROUND(AVG(t.speed_kmh), 2)                              AS avg_speed_kmh,
    ROUND(MAX(t.altitude_msl_ft), 0)                        AS max_altitude_msl_ft,
    MIN(r.rotor_id) || '-' || MAX(r.rotor_id)               AS rotor_id,
    r.R_m,
    r.P_max_kW                                              AS motor_ceiling_kw,
    COALESCE(r.notes, r.R_m || 'm / ' || r.P_max_kW || 'kW') AS rotor_variant,
    ROUND(SUM(CASE WHEN d.dt_s IS NOT NULL THEN d.kw * d.dt_s ELSE 0 END) / 1000.0, 2)
                                                            AS rotor_energy_MJ,
    ROUND(fp.rotor_fleet_max_power_kW, 1)                   AS rotor_fleet_max_power_kW,
    MIN(t.soc_pct)                                          AS soc_pct,
    MAX(t.fuel_kg)                                          AS fuel_kg
FROM test_parameters p
CROSS JOIN rotor_config r
JOIN rotor_kw_dt d      ON d.rotor_id    = r.rotor_id
JOIN telemetry t        ON t.timestamp_s = d.timestamp_s
JOIN fleet_peak_max fp  ON fp.R_m = r.R_m AND fp.P_max_kW = r.P_max_kW
GROUP BY r.R_m, r.P_max_kW;
