# The Douro Paradox: write-up

**Flag:** `SIGMOID_DISTANCE_FROM_PORTO_KM`

Code: `douro_paradox.py` (reproduces every number below). Data: `douro_vineyard_data.csv` (`;` separated, 3,543 plots).

## 1. Overall: treatment is ahead

| method    | plots | successes | success rate |
|-----------|------:|----------:|-------------:|
| control   | 1,653 |       447 |   **27.04%** |
| treatment | 1,890 |       585 |   **30.95%** |

Treatment leads by about 3.9 percentage points.

## 2. The column and the split

`distance_from_porto_km` is the only column whose plots separate cleanly into two groups.
No plot lies between 104 km and 114 km, so I cut at the middle of that gap (109 km):

* **near**: distance <= 109 km
* **far**: distance > 109 km

Any cutpoint inside the empty stretch gives identical groups.

| group (distance) | treatment plots | control plots | treatment rate | control rate | treatment - control |
|------------------|----------------:|--------------:|---------------:|-------------:|--------------------:|
| near (<= 109 km) |             144 |         1,233 |         25.00% |   **25.30%** |              -0.30 pp |
| far (> 109 km)   |           1,746 |           420 |         31.44% |   **32.14%** |              -0.70 pp |

Control is ahead in both groups, so the reversal is complete.

I screened every other column (categorical columns as they are, numeric columns in quartiles, plus all 2-group threshold splits).
None reverses in every group: cooperative 3/12, slope_aspect 1/4, vine_density 1/4, soil_ph 1/4 and so on.

## 3. Why pooling produces the reversal

Plots far from Porto are both much more likely to receive the new method (80.6% treated, versus 10.5% of near plots)
and naturally more successful (31.6% versus 25.3% baseline),
so the treatment group is mostly made of the easier far plots while the control group is mostly made of the harder near plots.
That mix, not the irrigation method, produces the apparent win.

## Caveat

The within-group gaps are small (0.3 and 0.7 percentage points) and I did not test them for statistical significance;
the claim is only that the direction reverses in every group.
