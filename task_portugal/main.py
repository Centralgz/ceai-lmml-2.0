import numpy as np
import pandas as pd

df = pd.read_csv("douro_vineyard_data.csv", sep=";")
df["treated"] = (df["treatment"] == "treatment").astype(int)

# ---------------------------------------------------------------- 1. overall
overall = df.groupby("treatment")["harvest_success"].agg(["mean", "sum", "size"])
print("OVERALL success rate\n", overall.round(4), "\n")

# ---------------------------------------------------------------- 2. screen every column
def n_groups_control_ahead(groups):
    ahead = total = 0
    for _, g in df.groupby(groups, observed=True):
        t = g.loc[g.treated == 1, "harvest_success"]
        c = g.loc[g.treated == 0, "harvest_success"]
        if len(t) == 0 or len(c) == 0:
            continue
        total += 1
        ahead += c.mean() > t.mean()
    return ahead, total

categorical = ["trellis_system", "vineyard_age_group", "cooperative", "grape_variety",
               "organic_certified", "harvest_month", "slope_aspect"]
numeric = ["soil_depth_cm", "sun_hours_july", "vine_density_per_ha", "annual_rainfall_mm",
           "distance_from_porto_km", "elevation_m", "soil_ph"]

print("SCREEN: groups where control beats treatment (full reversal = all groups)")
for col in categorical:
    a, t = n_groups_control_ahead(df[col])
    print(f"  {col:24s} {a}/{t}")
for col in numeric:
    a, t = n_groups_control_ahead(pd.qcut(df[col], 4, duplicates="drop"))
    print(f"  {col:24s} {a}/{t} (quartiles)")

# ---------------------------------------------------------------- 3. the culprit
col = "distance_from_porto_km"
d = df[col].sort_values().to_numpy()
gap_i = np.argmax(np.diff(d[(d > 90) & (d < 125)]))            # widest empty stretch near 100-120 km
vals = d[(d > 90) & (d < 125)]
lo, hi = vals[gap_i], vals[gap_i + 1]
cut = round((lo + hi) / 2, 1)
print(f"\nNo plots between {lo} and {hi} km -> natural cutpoint {cut} km")

df["distance_group"] = np.where(df[col] <= cut, f"near (<= {cut} km)", f"far (> {cut} km)")

print("\nSHARE OF PLOTS TREATED by group")
print(df.groupby("distance_group")["treated"].agg(["mean", "size"]).round(3))

rows = []
for g, sub in df.groupby("distance_group"):
    t = sub[sub.treated == 1]["harvest_success"]
    c = sub[sub.treated == 0]["harvest_success"]
    rows.append(dict(group=g, n_treatment=len(t), n_control=len(c),
                     rate_treatment=round(t.mean(), 4), rate_control=round(c.mean(), 4),
                     diff_treat_minus_ctrl=round(t.mean() - c.mean(), 4)))
res = pd.DataFrame(rows)
print("\nSUCCESS RATES WITHIN EACH GROUP\n", res.to_string(index=False))
assert (res.diff_treat_minus_ctrl < 0).all(), "not a full reversal"
print("\nControl is ahead in EVERY group -> complete Simpson's reversal.")

base = df.groupby("distance_group")["harvest_success"].mean().round(4)
print("\nBaseline success per group (all plots):\n", base)

print("\nFLAG: SIGMOID_DISTANCE_FROM_PORTO_KM")
res.to_csv("douro_reversal_table.csv", index=False)
