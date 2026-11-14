"""Call My BLEUff: separate the honest deck from the two BLEU-gaming ones.

The house scorer is corpus BLEU-4 with multiple references (see bleu.py):
an n-gram counts if it occurs in any reference, and the brevity penalty never
punishes a deck for being long. Two different ways of writing for that scorer
show up in the panel.

1. Cross-reference stitching. Paste a sentence from one reference onto a
   sentence from another. Every 4-gram is attested somewhere, the text is
   longer than either reference, and BLEU does not mind. A listener hears
   the venue described twice, sometimes with two different prices.

2. Omission. Keep only the spans that already occur in the references and
   drop the rest of the facts. Precision stays high because the unmatched
   words never enter the denominator, and a brevity penalty computed once
   per deck hides a few short cards.

Neither bluffer does it on every card, and they do not do it the same way.
The honest deck states the facts the references agree on, in its own
sentences, and does not glue two references together.

Score (lower is more honest), per deck:
    missed consensus facts
  + missed consensus venue names
  + 8 * cards that paste sentences taken from two different references
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bleu import tokenize

HERE = Path(__file__).resolve().parent

FAMILIES = {
    "price": {
        "cheap": r"cheap|inexpensive|low[- ]pric|low[- ]cost|\bbudget\b|affordabl|less than £?\s?20|under £?\s?20|below £?\s?20|less than 20|under 20|on a budget|price range is low|prices are low|range is low",
        "moderate": r"moderate(?:ly)?[- ]pric|mid[- ](?:range|pric)|medium[- ](?:pric|range)|£\s?20\s?[-–]\s?£?\s?25|£\s?20 to £?\s?25|reasonably priced|average[- ]pric|prices are average|average prices|average price|moderate amount|priced moderate",
        "high": r"expensive|high(?:ly)?[- ]pric|higher[- ]pric|high price|higher end|high[- ]end|priced high|more than £?\s?30|over £?\s?30|above £?\s?30|£30 and (?:over|up)|more than 30|pricey|costly|price range is high|prices are high|high price range|very expensive",
    },
    "rating": {
        "low": r"1 out of 5|1 out of five|one out of (?:5|five)|one[- ]star|1[- ]star|low(?:ly)?[- ]rat|rated (?:as )?low|rating is low|low reviews|poorly rated|poor rated|rated poorly|not rated well|bad rating|low customer|rating of low",
        "average": r"3 out of 5|3 out of five|three out of five|three[- ]star|3[- ]star|average(?: customer)? rat|average reviews|reviews are average|rating is average|rated average|moderately rated",
        "high": r"5 out of 5|5 out of five|five out of five|five[- ]star|5[- ]star|high(?:ly)?[- ]rat|highly (?:customer |very )?rated|rating of high|rating is high|rated (?:as )?high|given a rating of high|top[- ]rated|high customer|excellent",
    },
    "family": {
        "yes": r"family[- ]friend(?:ly)?|family oriented|family food|whole family|entire family|\bfamilies\b|child(?:ren)?[- ]friend(?:ly)?|kid(?:s)?[- ]friend(?:ly)?|friendly to (?:kids|children|families)|for (?:the )?(?:families|kids|children)|for famil|children (?:are |is )?welcome|kids (?:are |is )?welcome|welcomes (?:children|kids|families)|families are welcome|with your children|great for children|for children|suitable for families|good for (?:a )?famil|family (?:pub|restaurant|venue|place|coffee)",
        "no": r"not (?:a |an )?(?:very |so )?(?:family|child|kid)|isn'?t (?:a |an )?(?:very |so )?(?:family|child|kid)|no families|no children|no kids|adults only|not suitable for families|not recommended for (?:families|children|kids)|non[- ]family",
    },
    "food": {
        "japanese": r"japanese|sushi|oriental",
        "italian": r"\bitalian\b|spaghetti|pizza",
        "english": r"\benglish\b|\bbritish\b",
        "french": r"\bfrench\b",
        # "Raja Indian Cuisine" is a neighbouring venue, not the food served here.
        "indian": r"\bindian food\b|(?<!raja )\bindian cuisine\b|serves indian|serving indian|includes indian|indian restaurant|indian pub|indian eatery",
        "chinese": r"\bchinese\b",
        "fast": r"fast food|fast-food",
    },
    "eat": {
        "coffee": r"coffee shop|coffeehouse",
        "pub": r"\bpub\b|public house",
        "restaurant": r"\brestaurant\b",
    },
    "area": {
        "riverside": r"riverside|by the river|on the river|along the river|near the river|along ?side the river|close to the river|next to the river|banks of the river|by a river|river ?side|near a river|waterfront|waterside",
        "centre": r"city centre|city center|centre of the city",
    },
}

# Single words that the capitalisation heuristic mistakes for venue names.
NOT_A_NAME = {
    "coffee shop", "fast food", "city centre", "city center", "there",
    "riverside", "city", "english", "french", "italian", "japanese",
    "indian", "chinese", "yes", "near", "with", "this", "that",
}
NAME_STOP = {"The", "A", "An", "In", "On", "At", "Of", "For", "With", "And",
             "Near", "Restaurant", "Pub", "Coffee", "Shop"}
STITCH_PENALTY = 8


def find_values(text: str, family: dict) -> list[str]:
    low = text.lower()
    found = [val for val, pat in family.items() if re.search(pat, low)]
    if "no" in found and "yes" in found:
        found = [v for v in found if v != "yes"]
    return found


def consensus(texts, family, k: int) -> set[str]:
    counts: Counter = Counter()
    for text in texts:
        counts.update(set(find_values(text, family)))
    return {val for val, n in counts.items() if n >= k}


def candidate_names(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-zÀ-ÿ0-9'’£-]+|[^\w\s]", text)
    names = []
    i = 0
    while i < len(toks):
        word = toks[i]
        if word[:1].isupper():
            j = i
            buf = []
            while j < len(toks) and re.fullmatch(r"[A-Za-zÀ-ÿ0-9'’]+", toks[j]) and (
                toks[j][:1].isupper() or toks[j].lower() in {"of", "the", "and", "by"}
            ):
                buf.append(toks[j])
                j += 1
            if buf and not all(w in NAME_STOP for w in buf):
                names.append(" ".join(buf))
            i = j if j > i else i + 1
        else:
            i += 1
    return names


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def consensus_names(references: list[str]) -> list[str]:
    counts: Counter = Counter()
    for ref in references:
        seen = set()
        for name in candidate_names(ref):
            key = norm_name(name)
            if len(key) >= 4 and key not in NOT_A_NAME:
                seen.add(key)
        counts.update(seen)
    kept: list[str] = []
    for name, n in sorted(counts.items(), key=lambda kv: -len(kv[0])):
        if n >= 4 and not any(name != other and name in other for other in kept):
            kept.append(name)
    return kept


def is_span(sentence: list[str], reference: list[str]) -> bool:
    n, m = len(sentence), len(reference)
    if n == 0 or n > m:
        return False
    for i in range(m - n + 1):
        if reference[i:i + n] == sentence:
            return True
    return False


def pasted_sentences(text: str, references: list[str]) -> int:
    """Sentences of at least 5 tokens copied verbatim from some reference."""
    refs = [tokenize(r) for r in references]
    sentences = [tokenize(s) for s in text.replace("!", ".").replace("?", ".").split(".")]
    return sum(
        1 for sent in sentences
        if len(sent) >= 5 and any(is_span(sent, ref) for ref in refs)
    )


def cross_reference_stitch(text: str, references: list[str]) -> bool:
    """Two copied sentences that no single reference contains both of.

    Multi-reference BLEU clips each n-gram against whichever reference has
    it most often, and it never charges for extra length. Pasting one
    sentence from each of two references therefore scores like a perfect
    description while a listener hears the venue described twice. Two
    sentences copied from the same reference are just a close wording of
    one description, which an honest panellist does as well.
    """
    refs = [tokenize(r) for r in references]
    sentences = [tokenize(s) for s in text.replace("!", ".").replace("?", ".").split(".")]
    sources = []
    for sent in sentences:
        if len(sent) < 5:
            continue
        hits = {i for i, ref in enumerate(refs) if is_span(sent, ref)}
        if hits:
            sources.append(hits)
    if len(sources) < 2:
        return False
    return not set.intersection(*sources)


def decode(seats: list[int]) -> str:
    word = ""
    for i in range(0, len(seats), 3):
        digits = [s - 1 for s in seats[i:i + 3]]
        value = 9 * digits[0] + 3 * digits[1] + digits[2]
        word += chr(ord("A") + value)
    return word


def main() -> None:
    refs = pd.read_csv(HERE / "references.csv")
    panel = pd.read_csv(HERE / "panel.csv")
    ref_map = refs.groupby(["round", "card"]).reference.apply(list).to_dict()

    rows = []
    for _, row in panel.iterrows():
        references = ref_map[(row["round"], row.card)]
        text = row.text
        missed = 0
        for family in FAMILIES.values():
            missed += len(consensus(references, family, 4) - set(find_values(text, family)))
        low = re.sub(r"[^a-z0-9 ]", "", text.lower())
        missed += sum(1 for name in consensus_names(references) if name not in low)
        rows.append({
            "round": int(row["round"]),
            "seat": int(row.seat),
            "missed": missed,
            "stitched": int(cross_reference_stitch(text, references)),
        })
    cards = pd.DataFrame(rows)
    decks = cards.groupby(["round", "seat"], as_index=False).agg(
        missed=("missed", "sum"), stitched=("stitched", "sum")
    )
    decks["score"] = decks.missed + STITCH_PENALTY * decks.stitched

    winners = []
    print(f"{'rnd':>4}  {'seat1':>8} {'seat2':>8} {'seat3':>8}  honest  gap")
    for rnd, group in decks.groupby("round"):
        group = group.sort_values("seat")
        scores = {int(r.seat): r.score for r in group.itertuples()}
        order = sorted(scores, key=lambda s: (scores[s], s))
        honest = order[0]
        gap = scores[order[1]] - scores[honest]
        winners.append(honest)
        shown = "  ".join(f"{scores[s]:8.0f}" for s in (1, 2, 3))
        print(f"{rnd:4d}  {shown}  {honest:6d}  {gap:4.0f}")

    flag = "SIGMOID_" + decode(winners)
    print("\nhonest seats:", winners)
    print("flag:", flag)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for seat, group in decks.groupby("seat"):
        ax.scatter(group["round"], group.score, label=f"seat {seat}", s=30)
    honest_rows = decks.loc[decks.groupby("round").score.idxmin()]
    ax.plot(honest_rows["round"], honest_rows.score, color="0.3", lw=1, zorder=0)
    ax.set_xlabel("round")
    ax.set_ylabel("missed facts + 8 × cross-reference stitches")
    ax.set_xticks(range(1, 25))
    ax.set_ylim(-2, 90)
    ax.legend(frameon=False, ncol=3)
    off = decks[decks.score > 90]
    for row in off.itertuples():
        ax.annotate(
            f"seat {row.seat}: {row.score:.0f}",
            (row.round, 86),
            ha="center",
            fontsize=8,
            color="0.3",
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "honesty.png", dpi=130)
    print("figure: honesty.png")


if __name__ == "__main__":
    main()
