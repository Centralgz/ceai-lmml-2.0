"""Pipes, Leaks, and the Graph Beneath
Clean the raw ferry/train/bus route log, rebuild the real network and rank towns by
number of distinct connections.

Run:  python main.py        (needs routes.csv next to this file)
Writes: towns_ranking.csv, connections.csv, town_aliases.csv, network_top8.png
"""
import re
import unicodedata
from collections import defaultdict

import pandas as pd

RAW = pd.read_csv("routes.csv", encoding="utf-8")


# --------------------------------------------------------------------- 1. formatting noise
def fold(name: str) -> str:
    """Whitespace, trailing dots, upper/lower case, accents (a/ä, o/ö, å/a) -> one form."""
    s = unicodedata.normalize("NFKC", str(name))
    s = re.sub(r"[\s\.,;\-_]+", " ", s).strip()
    s = unicodedata.normalize("NFD", s.casefold())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# --------------------------------------------------------------------- 2. Swedish / old names
# Many Finnish towns are logged under their Swedish (or older) names by some offices.
# Every alias below points to the official Finnish name (folded form).
ALIAS = {
    "abo": "turku", "alavo": "alavus", "birkala": "pirkkala", "bjorneborg": "pori",
    "borga": "porvoo", "brahestad": "raahe", "esbo": "espoo", "fredrikshamn": "hamina",
    "grankulla": "kauniainen", "hango": "hanko", "helsingfors": "helsinki",
    "hogfors": "karkkila", "hyvinge": "hyvinkaa", "idensalmi": "iisalmi",
    "ikalis": "ikaalinen", "ilmola": "ilmajoki", "inga": "inkoo",
    "jakobstad": "pietarsaari", "kajana": "kajaani", "karleby": "kokkola",
    "kasko": "kaskinen", "kervo": "kerava", "keuru": "keuruu",
    "kimitoon": "kemionsaari", "korsholm": "mustasaari",
    "kristinestad": "kristiinankaupunki", "kronoby": "kruunupyy", "kumo": "kokemaki",
    "kyrkslatt": "kirkkonummi", "lahtis": "lahti", "lappo": "lapua", "larsmo": "luoto",
    "lojo": "lohja", "lovisa": "loviisa", "lundo": "lieto", "malax": "maalahti",
    "narpes": "narpio", "nadendal": "naantali", "nystad": "uusikaupunki",
    "nykarleby": "uusikaarlepyy", "pargas": "parainen", "raumo": "rauma",
    "raseborg": "raasepori", "reso": "raisio", "s:t karins": "kaarina",
    "s:t michel": "mikkeli", "sibbo": "sipoo", "sjundea": "siuntio",
    "tammerfors": "tampere", "tavastehus": "hameenlinna", "tornea": "tornio",
    "traskanda": "jarvenpaa", "tusby": "tuusula", "uleaborg": "oulu",
    "ulvsby": "ulvila", "vanda": "vantaa", "vasa": "vaasa", "vichtis": "vihti",
    "villmanstrand": "lappeenranta", "virmo": "mynamaki", "vora": "voyri",
}


def canonical(name: str) -> str:
    f = fold(name)
    return ALIAS.get(f, f)


df = RAW.copy()
df["a"] = df["from_town"].map(canonical)
df["b"] = df["to_town"].map(canonical)

# --------------------------------------------------------------------- 3. connections
n_before = len(df)
self_loops = (df.a == df.b).sum()
df = df[df.a != df.b]                                         # a route from a town to itself is not a connection
df["edge"] = [tuple(sorted(p)) for p in zip(df.a, df.b)]       # A-B == B-A
edges = sorted(set(df["edge"]))                                # every connection once
# (route_type and office are ignored: a ferry, a train and a bus between the same two
#  towns are still ONE connection between those two towns)

nbrs = defaultdict(set)
for a, b in edges:
    nbrs[a].add(b)
    nbrs[b].add(a)

ranking = (pd.Series({t: len(v) for t, v in nbrs.items()}, name="connections")
           .rename_axis("town").reset_index()
           .sort_values(["connections", "town"], ascending=[False, True])
           .reset_index(drop=True))

print(f"raw log entries      : {n_before}")
print(f"distinct raw names   : {pd.concat([RAW.from_town, RAW.to_town]).nunique()}")
print(f"self-loops removed   : {self_loops}")
print(f"towns after cleaning : {len(nbrs)}   (expected 116)")
print(f"connections          : {len(edges)}   (expected 448)")
print("\nTOP 10 TOWNS\n", ranking.head(10).to_string(index=False))

top8 = ranking.head(8)["connections"]
assert top8.is_unique, "top-8 degrees should all differ"

flag_word = "".join(t[0] for t in ranking.head(7)["town"]).upper()
print("\nFLAG: SIGMOID_" + flag_word)

# --------------------------------------------------------------------- 4. outputs
ranking.to_csv("towns_ranking.csv", index=False)
pd.DataFrame(edges, columns=["town_a", "town_b"]).to_csv("connections.csv", index=False)
raw_names = pd.concat([RAW.from_town, RAW.to_town]).drop_duplicates()
pd.DataFrame({"raw_name": raw_names, "clean_town": raw_names.map(canonical)}) \
    .sort_values("clean_town").to_csv("town_aliases.csv", index=False)

try:                                                              # optional picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = ranking.head(8)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["town"].str.title()[::-1], top["connections"][::-1], color="#3a7d44")
    ax.set_xlabel("distinct connections")
    ax.set_title("Busiest towns in the cleaned network")
    fig.tight_layout()
    fig.savefig("network_top8.png", dpi=150)
except ImportError:
    pass
