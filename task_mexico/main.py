import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# Street-type abbreviations used by the registration desks.
# Only the short form is rewritten; the full word is left as-is.
ABBREVIATIONS = {
    "peat": "peatonal",
    "and": "andador",
    "ret": "retorno",
    "av": "avenida",
    "cor": "corredor",
    "priv": "privada",
    "blvd": "boulevard",
    "int": "interior",
    "edif": "edificio",
    "depto": "departamento",
    "ampl": "ampliacion",
    "circ": "circuito",
    "calz": "calzada",
    "diag": "diagonal",
    "prol": "prolongacion",
    "cont": "continuacion",
    "pas": "pasaje",
    "viad": "viaducto",
    "cerr": "cerrada",
}

# "street, city, STATE ZIP" — state may be "Q. ROO".
ADDRESS_RE = re.compile(
    r"^(?P<street>.*?),\s*(?P<city>.*?),\s*"
    r"(?P<state>[A-Z](?:[A-Z.]|\s(?=[A-Z]))*[A-Z.]?)\s+"
    r"(?P<zip>\d{5}(?:-\d{4})?)\s*$"
)
CALLEJON_RE = re.compile(r"(?<![a-z])c\.?\s*jon\.?")
CALLE_RE = re.compile(r"(?<![a-z])c\.")


def strip_accents(text):
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def canon_street(street):
    """Order-independent street key.

    Desks reorder the house numbers, drop accents, and shorten the street
    type (Calle / C., Callejón / C.jon / Cjón., Interior / Int., and so on).
    """
    text = strip_accents(street).lower()
    text = CALLEJON_RE.sub(" callejon ", text)
    text = CALLE_RE.sub(" calle ", text)
    text = text.replace(".", " ")
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text):
        token = ABBREVIATIONS.get(token, token)
        if token.isdigit():
            token = token.lstrip("0") or "0"
        tokens.append(token)
    return tuple(sorted(tokens))


def canon_address(address):
    match = ADDRESS_RE.match(address.strip())
    if match is None:
        # Keep an unmatched address as its own key so it cannot merge
        # with a different record by accident.
        return ("raw", strip_accents(address).lower())
    state = re.sub(r"[.\s]", "", match.group("state"))
    city = strip_accents(match.group("city")).lower()
    city = re.sub(r"\s+", " ", city).strip()
    return (canon_street(match.group("street")), city, state, match.group("zip"))


def find_matches(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[canon_address(row["address"])].append(row["record_id"])

    matches = []
    for record_ids in groups.values():
        unique_ids = sorted(set(record_ids))
        for left, right in combinations(unique_ids, 2):
            matches.append([left, right])
    matches.sort()
    return matches


def resolve_paths():
    if len(sys.argv) >= 3:
        return sys.argv[1], sys.argv[2]
    if len(sys.argv) == 2:
        return sys.argv[1], "test-output.json"
    if Path("test-input.csv").is_file():
        return "test-input.csv", "test-output.json"
    return "train-input.csv", "test-output.json"


def main():
    source, destination = resolve_paths()
    with open(source, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    payload = {"matches": find_matches(rows)}
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
