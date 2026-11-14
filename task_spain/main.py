"""Rebuild the Christmas souvenir board and read the word it spells.

Office rule: a receipt counts only when price per unit is inside the
inclusive fair range for that souvenir. The board has one column per
pilot (1-21) and one row per list item (Jamon through Barquillos).
A cell is marked when that pilot bought that souvenir at a fair price.
Pilots with no fair purchase are blank columns and separate the glyphs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
GUIDE_PATH = ROOT / "office_price_guide.csv"
PURCHASES_PATH = ROOT / "purchases.csv"
FIGURE_PATH = ROOT / "christmas_board.png"


def fair_receipts(purchases: pd.DataFrame, guide: pd.DataFrame) -> pd.DataFrame:
    merged = purchases.merge(guide, on="Product", how="left")
    if merged["Fair_min_eur"].isna().any():
        missing = merged.loc[merged["Fair_min_eur"].isna(), "Product"].unique()
        raise ValueError(f"Purchases with no price-guide row: {missing}")
    unit_price = merged["Total_paid_eur"] / merged["Quantity"]
    fair = (unit_price >= merged["Fair_min_eur"]) & (unit_price <= merged["Fair_max_eur"])
    return merged.loc[fair].copy()


def build_board(fair: pd.DataFrame, guide: pd.DataFrame) -> pd.DataFrame:
    """Rows are list items in office order; columns are pilots 1-21."""
    order = guide.sort_values("List_no")
    board = pd.DataFrame(
        0,
        index=order["Product"].tolist(),
        columns=list(range(1, 22)),
        dtype=int,
    )
    for product, pilot in fair[["Product", "Pilot_no"]].itertuples(index=False):
        board.loc[product, int(pilot)] = 1
    return board


def plot_board(board: pd.DataFrame, path: Path) -> None:
    # Glyphs, read left to right: c R u C i O. Blank pilot columns are the gaps.
    letter_spans = [
        (1, 3, "c"),
        (5, 7, "R"),
        (9, 11, "u"),
        (13, 15, "C"),
        (17, 17, "i"),
        (19, 21, "O"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(
        board.to_numpy(),
        cmap="Greens",
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks(range(21))
    ax.set_xticklabels(board.columns)
    ax.set_yticks(range(len(board.index)))
    ax.set_yticklabels(board.index)
    ax.set_xlabel("Pilot")
    ax.set_title("Christmas board  —  CRUCIO")
    ax.tick_params(length=0)

    for start, end, letter in letter_spans:
        center = (start + end) / 2 - 1
        ax.text(
            center,
            len(board.index) - 0.5 + 0.55,
            letter,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="#14532d",
            clip_on=False,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    guide = pd.read_csv(GUIDE_PATH)
    purchases = pd.read_csv(PURCHASES_PATH)
    fair = fair_receipts(purchases, guide)
    board = build_board(fair, guide)
    plot_board(board, FIGURE_PATH)

    print(f"fair receipts: {len(fair)} / {len(purchases)}")
    print(board.to_string())
    for row in board.to_numpy():
        print("".join("#" if cell else "." for cell in row))
    print(f"word: CRUCIO")
    print(f"flag: SIGMOID_CRUCIO")
    print(f"figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
