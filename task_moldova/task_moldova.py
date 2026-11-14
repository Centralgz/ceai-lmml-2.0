import numpy as np
import pandas as pd
from collections import Counter

OBS = "observations.csv"
REG = "ring_register.csv"
SPECIES = "Ciconia ciconia"
MIN_READS = 4
FAR_KM = 30

obs = pd.read_csv(OBS, dtype=str)
reg = pd.read_csv(REG, dtype=str)
for c in ("latitude", "longitude"):
    obs[c] = obs[c].astype(float)
for c in ("ringed_latitude", "ringed_longitude"):
    reg[c] = reg[c].astype(float)

reads = obs[obs.ring_read.notna() & (obs.species == SPECIES)].copy()
reads = reads.drop_duplicates(subset=["checklist_id", "ring_read"])
known = reg[reg.species == SPECIES]


def km(la1, lo1, la2, lo2):
    return np.hypot((la1 - la2) * 111.0, (lo1 - lo2) * 77.0)


def vote(ring_reads, weights=None):
    out = ""
    for p in range(6):
        c = Counter()
        for i, s in enumerate(ring_reads):
            if s[p] != "?":
                c[s[p]] += 1 if weights is None else weights[i]
        out += c.most_common(1)[0][0] if c else "?"
    return out


reads["site"] = reads.latitude.round(2).astype(str) + "," + reads.longitude.round(2).astype(str)

candidates = []
for site, g in reads.groupby("site"):
    if len(g) < MIN_READS:
        continue
    la, lo = g.latitude.mean(), g.longitude.mean()
    nearest = min(
        km(la, lo, a, b) for a, b in zip(known.ringed_latitude, known.ringed_longitude)
    )
    candidates.append((nearest, site, g))

candidates.sort(key=lambda x: -x[0])
for nearest, site, g in candidates[:5]:
    print(f"{site:14s} reads={len(g):3d} nearest_registered_stork_km={nearest:6.1f} consensus={vote(g.ring_read.tolist())}")

nearest, site, g = candidates[0]
assert nearest > FAR_KM

simple = vote(g.ring_read.tolist())

g2 = g.copy()
g2["obs_key"] = g2.observers.str.split("|").str[0]
per_obs = {}
for key, grp in g2.groupby("obs_key"):
    per_obs[key] = vote(grp.ring_read.tolist())
by_observer = vote(list(per_obs.values()))

by_day = vote(g.drop_duplicates(subset=["date"]).ring_read.tolist())

print("simple vote       :", simple)
print("one vote/observer :", by_observer)
print("one vote/date     :", by_day)

for p in range(6):
    c = Counter(s[p] for s in g.ring_read if s[p] != "?")
    print(p, c.most_common())

print("FLAG: SIGMOID_" + simple)