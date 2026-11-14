"""Streaming anomaly detector for bunker environmental sensors.

Scores each reading as it arrives. Each gallery keeps its own baseline:
a long rolling median, which a short spike or a brief drift cannot move,
and a short window that makes a sustained shift visible.
"""

from __future__ import annotations

from collections import deque

import numpy as np

FEATURES = (
    "temperature_c",
    "humidity_pct",
    "vibration_g",
    "acoustic_db",
    "air_pressure_kpa",
    "co2_ppm",
)

# Normal operating point of the galleries, used as a prior so the first
# readings can be scored before a local baseline has been learned.
PRIOR_MEAN = np.array(
    [13.003514, 72.018699, 0.119853, 32.098245, 101.503318, 510.346524],
    dtype=float,
)
PRIOR_STD = np.array(
    [0.372455, 0.580589, 0.010043, 0.349551, 0.099266, 10.525559],
    dtype=float,
)

POINT_FLAG = 3.55
DRIFT_W = 6
DRIFT_FLAG = 2.50
COLL_Z = 2.15
COLL_K = 3
# Long window for the baseline. A drift of about a minute-block stays a
# minority of this window, so the median does not follow it, but a real
# change of operating point becomes the majority and is learned.
BASE_W = 280
GUARD = 16
MIN_CORE = 48
SCORE_WIDTH = 0.30


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        e = np.exp(-x)
        return float(1.0 / (1.0 + e))
    e = np.exp(x)
    return float(e / (1.0 + e))


def _score_from(value: float, flag_at: float) -> float:
    """Map a deviation to (0, 1) so values at flag_at sit just above 0.5."""
    return _sigmoid((value - flag_at) / SCORE_WIDTH + 0.08)


class _ZoneState:
    def __init__(self) -> None:
        self.hist: deque[np.ndarray] = deque(maxlen=BASE_W)
        self.recent: deque[np.ndarray] = deque(maxlen=DRIFT_W)
        self.last_tick = -10_000
        self.prev_score = 0.0

    def _baseline(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.hist) < MIN_CORE:
            return PRIOR_MEAN, PRIOR_STD
        core = np.stack(self.hist, axis=0)
        if len(core) > GUARD + MIN_CORE:
            core = core[:-GUARD]
        center = np.median(core, axis=0)
        mad = np.median(np.abs(core - center), axis=0)
        mad_std = 1.4826 * mad
        # Keep the known sensor scale, but widen it if this gallery is noisier.
        # A short drift is a minority of the window, so it barely moves the MAD.
        std = np.clip(np.maximum(PRIOR_STD, mad_std), PRIOR_STD, 2.0 * PRIOR_STD)
        return center, std

    def update(self, x: np.ndarray, tick: int) -> float:
        if tick - self.last_tick > 3:
            self.recent.clear()
            self.prev_score = 0.0
        self.last_tick = tick

        baseline, std = self._baseline()
        z = (x - baseline) / std
        az = np.abs(z)
        max_z = float(az.max())

        self.recent.append(x.copy())
        med_z = 0.0
        coll_med = 0.0
        if len(self.recent) >= DRIFT_W:
            recent = np.median(np.stack(self.recent, axis=0), axis=0)
            med_abs = np.abs((recent - baseline) / std)
            med_z = float(med_abs.max())
            if int((med_abs >= COLL_Z).sum()) >= COLL_K:
                coll_med = float(np.sort(med_abs)[-COLL_K:].mean())

        n_moderate = int((az >= COLL_Z).sum())
        coll_now = float(np.sort(az)[-COLL_K:].mean()) if n_moderate >= COLL_K else 0.0

        point_s = _score_from(max_z, POINT_FLAG)
        drift_s = _score_from(med_z, DRIFT_FLAG) if med_z > 0.0 else 0.0
        coll_strength = max(coll_now, coll_med)
        coll_s = _score_from(coll_strength, COLL_Z) if coll_strength > 0.0 else 0.0
        raw = max(point_s, drift_s, coll_s)
        # A single soft sample inside a run of strong alarms is still part of it.
        score = max(raw, 0.56) if self.prev_score >= 0.92 and raw >= 0.40 else raw
        self.prev_score = raw

        self.hist.append(x.copy())

        if not np.isfinite(score):
            return 0.0
        return float(min(1.0, max(0.0, score)))


class Detector:
    def __init__(self) -> None:
        self._zones: dict[str, _ZoneState] = {}
        self._tick = 0

    def update(self, reading: dict) -> float:
        self._tick += 1
        zone = str(reading.get("bunker_zone", "_"))
        values = []
        for feature, mean in zip(FEATURES, PRIOR_MEAN):
            raw = reading.get(feature, mean)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = float(mean)
            if not np.isfinite(value):
                value = float(mean)
            values.append(value)
        x = np.asarray(values, dtype=float)
        state = self._zones.get(zone)
        if state is None:
            state = _ZoneState()
            self._zones[zone] = state
        return state.update(x, self._tick)
