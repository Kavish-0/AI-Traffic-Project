"""Long-horizon synthetic traffic time series in the project's TrafficFeatures schema.

Why: the forecasting service consumes the camera pipeline's own output
(speed, density, occupancy, queue, flow, ...), but weeks of such data do not
exist yet, and running the frame-level microsimulation for weeks is too slow.
This generator produces 1-minute windows directly with a macroscopic model:

    demand(t)   = approach base x daily profile (weekday double peak / weekend / holiday)
                  x day-to-day factor x AR(1) noise x rain factor
    capacity(t) = lanes x saturation flow x green ratio x rain factor x incident factor
    queue       : point queue  Q[t+1] = max(0, Q[t] + arrivals - departures)
    observables : flow = departures, inflow = arrivals, queue length = Q/lanes x jam spacing
                  (capped at the camera view), density / occupancy / space-mean speed from
                  queued + moving vehicles in the view (fundamental-diagram style)
    sensor data : measurement noise, random missing minutes, occasional outages

Weather is REAL (Open-Meteo archive for the site) when available, so rain
timing is realistic; incidents are random capacity drops. Everything is seeded.
The data are synthetic: they test the forecasting pipeline and give the
service a schema-compatible model, but they are not evidence about real traffic
(that is what the PeMS08 benchmark is for).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class ApproachSpec(BaseModel):
    series_id: str
    lanes: int = Field(2, ge=1)
    free_flow_kmh: float = 50.0
    peak_demand_ratio: float = 1.05       # peak demand / capacity (>1 -> queues build)
    am_weight: float = 1.0                # relative size of the morning peak
    pm_weight: float = 1.0


class SyntheticSeriesConfig(BaseModel):
    start: str = "2025-06-02T00:00:00+05:30"
    days: int = Field(42, ge=2)
    resolution_s: int = 60
    seed: int = 7
    tz: str = "Asia/Kolkata"
    saturation_flow_veh_h_lane: float = 1800.0
    green_ratio: float = 0.45
    view_length_m: float = 60.0           # camera ROI length (queues beyond it are not visible)
    jam_spacing_m: float = 7.0
    incident_rate_per_day: float = 0.25
    missing_minute_prob: float = 0.003
    outage_rate_per_day: float = 0.07
    approaches: list[ApproachSpec] = [
        ApproachSpec(series_id="junction_a_north", lanes=2, am_weight=1.25, pm_weight=0.8),
        ApproachSpec(series_id="junction_a_east", lanes=2, peak_demand_ratio=0.95),
        ApproachSpec(series_id="junction_a_south", lanes=2, am_weight=0.8, pm_weight=1.3),
        ApproachSpec(series_id="junction_a_west", lanes=1, free_flow_kmh=40.0, peak_demand_ratio=0.9),
    ]


def _profile(hour: np.ndarray, weekend: np.ndarray, am: float, pm: float) -> np.ndarray:
    def g(mu: float, s: float) -> np.ndarray:
        return np.asarray(np.exp(-(((hour - mu) / s) ** 2)))

    weekday = 0.08 + am * 0.95 * g(8.75, 1.1) + 0.35 * g(13.0, 2.5) + pm * 1.0 * g(18.0, 1.3)
    wkend = 0.08 + 0.55 * g(12.5, 3.5) + 0.35 * g(19.0, 2.0)
    return np.asarray(np.where(weekend, wkend, weekday) / max(am, pm, 1.0))


@dataclass
class _State:
    queue: float = 0.0
    noise: float = 0.0


def generate_series(cfg: SyntheticSeriesConfig, weather_hourly: pd.DataFrame | None = None,
                    holiday_country: str | None = "IN", with_labels: bool = False) -> pd.DataFrame:
    """Long-format frame: series_id, timestamp (UTC), TrafficFeatures columns, weather columns.

    ``with_labels`` adds ground truth for anomaly evaluation (the random sequence, hence every
    other column, is identical with or without it):
        gt_incident     1 while an injected capacity drop is active
        gt_load_ratio   demand / available capacity in that minute (> 1: a queue must grow)
    """
    rng = np.random.default_rng(cfg.seed)
    step_min = cfg.resolution_s / 60.0
    start = pd.Timestamp(cfg.start).tz_convert("UTC")
    n = int(cfg.days * 1440 / step_min)
    times = pd.date_range(start, periods=n, freq=f"{cfg.resolution_s}s")
    local = times.tz_convert(cfg.tz)
    hour = np.asarray(local.hour + local.minute / 60.0, dtype=float)
    weekend = np.asarray(local.dayofweek >= 5)
    if holiday_country:
        import holidays

        cal = holidays.country_holidays(holiday_country, years=sorted(set(local.year)))
        weekend = weekend | np.array([d in cal for d in local.date])

    if weather_hourly is not None and len(weather_hourly):
        from traffic_intel.forecasting.data.context import weather_to_grid

        weather = weather_to_grid(weather_hourly, times)
    else:
        log.warning("synthetic series: no weather supplied -> generating synthetic rain")
        weather = _synthetic_weather(times, rng)
    rain = weather["rain_mm"].fillna(0.0).to_numpy()
    rain_speed = 1.0 - np.minimum(0.20, 0.04 * rain)
    rain_cap = 1.0 - np.minimum(0.15, 0.03 * rain)
    rain_dem = 1.0 - np.minimum(0.10, 0.02 * rain)

    days = np.floor((np.arange(n) * step_min) / 1440).astype(int)
    frames = []
    for a in cfg.approaches:
        cap_h = a.lanes * cfg.saturation_flow_veh_h_lane * cfg.green_ratio
        day_factor = rng.lognormal(0.0, 0.06, size=cfg.days + 1)[days]
        demand_h = cap_h * a.peak_demand_ratio * _profile(hour, weekend, a.am_weight, a.pm_weight)
        demand_h = demand_h * day_factor * rain_dem

        # incidents: capacity drops
        inc = np.ones(n)
        for _ in range(rng.poisson(cfg.incident_rate_per_day * cfg.days)):
            s0 = rng.integers(0, n)
            dur = int(rng.uniform(10, 45) / step_min)
            inc[s0:s0 + dur] = rng.uniform(0.4, 0.7)
        cap_step = cap_h * rain_cap * inc * step_min / 60.0

        st = _State()
        out = np.zeros((n, 11))
        load = np.zeros(n)
        view_q_cap = a.lanes * cfg.view_length_m / cfg.jam_spacing_m
        for t in range(n):
            st.noise = 0.8 * st.noise + rng.normal(0, 0.06)
            lam = max(demand_h[t] * np.exp(st.noise), 0.0) * step_min / 60.0
            arrivals = rng.poisson(lam)
            capacity = rng.poisson(cap_step[t])
            departures = min(st.queue + arrivals, capacity)
            st.queue = max(0.0, st.queue + arrivals - departures)

            x = min(1.3, (lam / max(cap_step[t], 1e-6)))
            load[t] = lam / max(cap_step[t], 1e-6)
            v_move = a.free_flow_kmh * rain_speed[t] * 0.8 * (1.0 - 0.45 * min(x, 1.0) ** 2)
            q_view = min(st.queue, view_q_cap)
            queue_len = q_view / a.lanes * cfg.jam_spacing_m
            moving_len = max(cfg.view_length_m - queue_len, 5.0)
            moving = departures * (60.0 / step_min) / max(v_move, 1.0) * (moving_len / 1000.0)
            present = q_view + moving
            speed = (moving * v_move + q_view * 0.5) / max(present, 1e-6) if present > 0.05 else v_move
            density = present / (cfg.view_length_m / 1000.0 * a.lanes)
            out[t] = [
                arrivals + present,                          # vehicle_count (present + entering)
                max(0.0, speed * rng.normal(1.0, 0.04)),     # avg_speed_kmh (measurement noise)
                density,                                     # density_veh_km (per lane)
                min(150.0, density * 1.1) / 150.0,           # density_ratio (PCU ~ 1.1)
                min(98.0, density * 0.55 * rng.normal(1.0, 0.05)),  # occupancy_pct
                departures * 60.0 / step_min,                # flow_veh_h
                arrivals,                                    # inflow
                departures,                                  # outflow
                queue_len,                                   # queue_length_m
                round(q_view * 0.7),                         # stopped_vehicles
                1.0,                                         # quality
            ]
        df = pd.DataFrame(out, columns=[
            "vehicle_count", "avg_speed_kmh", "density_veh_km", "density_ratio", "occupancy_pct",
            "flow_veh_h", "inflow", "outflow", "queue_length_m", "stopped_vehicles", "quality"])
        df.insert(0, "timestamp", times)
        df.insert(0, "series_id", a.series_id)
        if with_labels:
            df["gt_incident"] = (inc < 1.0).astype(int)
            df["gt_load_ratio"] = np.round(load, 3)

        # sensor problems: isolated missing minutes and longer outages -> NaN rows
        miss = rng.random(n) < cfg.missing_minute_prob
        for _ in range(rng.poisson(cfg.outage_rate_per_day * cfg.days)):
            s0 = rng.integers(0, n)
            miss[s0:s0 + int(rng.uniform(20, 120) / step_min)] = True
        traffic_cols = [c for c in df.columns[2:] if not c.startswith("gt_")]
        df.loc[miss, traffic_cols] = np.nan
        for c in weather.columns:
            df[c] = weather[c].to_numpy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _synthetic_weather(times: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(times)
    hour = times.hour + times.minute / 60.0
    temp = 27 + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.5, n)
    rain = np.zeros(n)
    t = 0
    while t < n:
        t += int(rng.exponential(1440 * 1.5))
        dur = int(rng.uniform(30, 240))
        rain[t:t + dur] = rng.gamma(2.0, 1.5)
        t += dur
    return pd.DataFrame({
        "temperature_c": temp, "humidity_pct": np.clip(70 + 3 * rain + rng.normal(0, 5, n), 20, 100),
        "rain_mm": rain, "visibility_km": np.nan,
        "weather_condition": np.where(rain > 0.2, "rain", "cloudy"),
    }, index=times)
