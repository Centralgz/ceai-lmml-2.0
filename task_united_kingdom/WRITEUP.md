# Call My BLEUff

**Flag: `SIGMOID_FRINDLOW`**

The producers crowned the highest corpus BLEU in each round and spelled `SIGMOID_SKELDRAP`. That word is what the house scorer was gamed into saying. The honest panellist is the one a listener can follow. Naming that seat in every round spells `FRINDLOW`.

## What this BLEU rewards

`bleu.py` is corpus BLEU-4, and three choices in it are the whole attack surface.

- An output n-gram counts if it occurs in **any** of the five references. Clipping is the maximum count inside a single reference, so a sentence taken from reference A and a sentence taken from reference B both count in full.
- There is no smoothing. Precision is a geometric mean, but a deck that stays inside the reference phrasing never hits a zero.
- The brevity penalty is computed **once per deck**, from the summed length against the summed closest-reference length, and a deck that runs long pays nothing. Ties for the closest reference go to the shorter one, so a short extract is also cheap.

A description can therefore score like a good one while saying the venue twice, or while saying almost nothing.

## How the bluffers play it

They do not bluff every card, and the two of them in a round do not bluff the same way. Both ways are aimed at the scorer above.

**Cross-reference stitching.** Paste one reference sentence onto a sentence from a different reference. Every 4-gram is attested, the card is longer than either source, and BLEU does not mind. Round 11, card 27, seat 3:

> Fitzbillies is a 5 star rated, child friendly coffee shop in the riverside area that serves expensive French food. Fitzbillies is an expensive, family-friendly coffee shop located in the City Centre.

Those are two references verbatim. One puts Fitzbillies in riverside, the other in the City Centre. The listener hears both. Seat 3 does this on 4 cards in that round and on 26 cards in round 23, where the deck's BLEU is 0.784, the highest on the show.

**Omission.** Keep a span the references already contain and drop the rest of the facts. Unmatched words never enter the precision denominator, and a handful of short cards cannot move a corpus brevity penalty. Round 11, card 20, seat 1:

> Loch Fyne is low customer rated restaurant.

The five references agree that Loch Fyne is a low-rated French restaurant by the river, near The Rice Boat. Seat 2, the honest one that round, says so. Seat 1's seven-word extract is what BLEU prefers: that deck scores 0.551 and **wins the round**, while the honest deck scores 0.407, the lowest of the three.

The same thing, smaller, is round 5 card 13, seat 1: "Near Ranch is The Cricketers." The pub, the low rating and the "not family-friendly" are gone. Seat 2 of that round is the stitcher instead (nine cards), so the honest speaker is seat 3.

## The check

A venue fact counts as agreed when at least four of the five references state it. The attributes are the ones the guide actually varies: price, rating, family-friendliness, food, kind of place, area, plus a venue name that four references share. Wording counts. "Five out of five", "high-end", "next to the river", "friendly to kids", "£20 to £25" and "the reviews are average" are the same facts as the shorter labels. "Raja Indian Cuisine" does not count as the food being Indian; it is the place next door.

For one card the penalty is

- one point for each agreed fact, or agreed name, that the card does not state, plus
- 8 if the card pastes two sentences of at least five tokens that **no single reference contains both of**.

A deck's score is the sum over its 40 cards. Lower is more honest. Quoting two sentences of the *same* reference is not the penalty: that is one description, and an honest speaker sometimes lands on it. The 8 is large enough that a stitcher who remembered the facts still loses, and small enough that one echoed sentence does not drown a real omission.

`python solution.py` prints the table and writes `honesty.png`. In every round the lowest score is at least 3 points clear of the other two. Ten of the honest decks miss nothing.

| Round | Seat 1 | Seat 2 | Seat 3 | Honest | Gap |
|------:|-------:|-------:|-------:|:------:|----:|
| 1 | 1 | 13 | 33 | 1 | 12 |
| 2 | 13 | 1 | 43 | 2 | 12 |
| 3 | 37 | 8 | 1 | 3 | 7 |
| 4 | 11 | 0 | 26 | 2 | 11 |
| 5 | 7 | 74 | 1 | 3 | 6 |
| 6 | 43 | 17 | 0 | 3 | 17 |
| 7 | 0 | 8 | 31 | 1 | 8 |
| 8 | 8 | 38 | 1 | 3 | 7 |
| 9 | 61 | 7 | 0 | 3 | 7 |
| 10 | 33 | 0 | 11 | 2 | 11 |
| 11 | 7 | 0 | 32 | 2 | 7 |
| 12 | 9 | 2 | 193 | 2 | 7 |
| 13 | 1 | 11 | 75 | 1 | 10 |
| 14 | 58 | 1 | 10 | 2 | 9 |
| 15 | 0 | 13 | 50 | 1 | 13 |
| 16 | 11 | 0 | 37 | 2 | 11 |
| 17 | 0 | 3 | 29 | 1 | 3 |
| 18 | 48 | 14 | 1 | 3 | 13 |
| 19 | 45 | 3 | 11 | 2 | 8 |
| 20 | 47 | 3 | 15 | 2 | 12 |
| 21 | 7 | 35 | 3 | 3 | 4 |
| 22 | 67 | 14 | 2 | 3 | 12 |
| 23 | 15 | 2 | 208 | 2 | 13 |
| 24 | 42 | 5 | 11 | 2 | 6 |

The two scores off the top of the figure are round 12 seat 3 (193) and round 23 seat 3 (208): almost every card is two references glued together. The line along the bottom is the honest seat.

The closest call is round 17, and it is not close once the cards are read. Seat 1 states every agreed fact. Seat 2 drops "family-friendly" for the Green Man, drops The Portland Arms for the Cricketers, and drops the low rating for The Punter. Seat 3 is fragmentary on most of the deck.

## The word

Seat 1 is the digit 0, seat 2 is 1, seat 3 is 2. Three rounds are one letter: `9·d1 + 3·d2 + d3`, with 0 = A.

Honest seats, in order:

`1 2 3 · 2 3 3 · 1 3 3 · 2 2 2 · 1 2 1 · 2 1 3 · 2 2 3 · 3 2 2`

| Rounds | Digits | Value | Letter |
|--------|--------|------:|:------:|
| 1–3 | 0, 1, 2 | 5 | F |
| 4–6 | 1, 2, 2 | 17 | R |
| 7–9 | 0, 2, 2 | 8 | I |
| 10–12 | 1, 1, 1 | 13 | N |
| 13–15 | 0, 1, 0 | 3 | D |
| 16–18 | 1, 0, 2 | 11 | L |
| 19–21 | 1, 1, 2 | 14 | O |
| 22–24 | 2, 1, 1 | 22 | W |

`SIGMOID_FRINDLOW`
