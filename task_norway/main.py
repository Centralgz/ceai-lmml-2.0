"""Put back the digits the office nisse swapped on its bad nights.

On a bad night the nisse exchanges one pair of digits every time it writes
them, and leaves every other digit alone. The training archive has the true
labels, so a classifier trained on those labels recognizes the scans. Where
its reading and the filed label disagree as a reciprocal pair, inside one
contiguous run of sample ids, that run is a bad night: swap the pair back.
"""

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
TRAIN_INPUT = ROOT / "train-input.csv"
TRAIN_OUTPUT = ROOT / "train-output.csv"
TEST_INPUT = ROOT / "test-input.csv"
TEST_OUTPUT = ROOT / "test-output.csv"
MODEL_PATH = ROOT / "model.joblib"

MIN_EACH_DIRECTION = 4


def pixel_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("pixel_")]


def sample_number(sample_id: pd.Series) -> pd.Series:
    return sample_id.str.extract(r"(\d+)", expand=False).astype(int)


def reciprocal_pairs(given: np.ndarray, predicted: np.ndarray) -> list[tuple[int, int]]:
    """Digit pairs the nisse may have swapped: both directions, many times."""
    counts: Counter[tuple[int, int]] = Counter()
    for filed, read in zip(given, predicted):
        if filed != read:
            counts[(int(filed), int(read))] += 1
    pairs = []
    seen = set()
    for (left, right), _count in counts.items():
        pair = tuple(sorted((left, right)))
        if pair in seen or left == right:
            continue
        seen.add(pair)
        forward = counts[(pair[0], pair[1])]
        backward = counts[(pair[1], pair[0])]
        if forward >= MIN_EACH_DIRECTION and backward >= MIN_EACH_DIRECTION:
            pairs.append(pair)
    return pairs


def bad_night_spans(
    sample_ids: np.ndarray,
    given: np.ndarray,
    predicted: np.ndarray,
    first: int,
    second: int,
) -> list[tuple[int, int]]:
    """Spans of sample ids with no correctly filed digit of the swapped pair inside."""
    order = np.argsort(sample_ids, kind="mergesort")
    swaps: list[int] = []
    intact: list[int] = []
    for index in order:
        filed = int(given[index])
        read = int(predicted[index])
        number = int(sample_ids[index])
        if {filed, read} == {first, second} and filed != read:
            swaps.append(number)
        elif filed == read and filed in (first, second):
            intact.append(number)
    if len(swaps) < MIN_EACH_DIRECTION:
        return []

    groups: list[list[int]] = [[swaps[0]]]
    for number in swaps[1:]:
        previous = groups[-1][-1]
        if any(previous < kept < number for kept in intact):
            groups.append([number])
        else:
            groups[-1].append(number)
    return [(min(group), max(group)) for group in groups]


def restore_labels(
    sample_ids: np.ndarray,
    given: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    corrected = given.copy()
    for first, second in reciprocal_pairs(given, predicted):
        for start, end in bad_night_spans(sample_ids, given, predicted, first, second):
            span = (sample_ids >= start) & (sample_ids <= end)
            filed_first = span & (corrected == first)
            filed_second = span & (corrected == second)
            corrected[filed_first] = second
            corrected[filed_second] = first
    return corrected


def main() -> None:
    train_in = pd.read_csv(TRAIN_INPUT)
    train_out = pd.read_csv(TRAIN_OUTPUT)
    test = pd.read_csv(TEST_INPUT)
    pixels = pixel_columns(train_in)

    train = train_in.merge(train_out, on="sample_id")
    classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))
    classifier.fit(train[pixels], train["corrected_label"])
    joblib.dump(classifier, MODEL_PATH)

    predicted = classifier.predict(test[pixels])
    given = test["given_label"].to_numpy()
    sample_ids = sample_number(test["sample_id"]).to_numpy()
    corrected = restore_labels(sample_ids, given, predicted)

    result = pd.DataFrame(
        {"sample_id": test["sample_id"], "corrected_label": corrected.astype(int)}
    )
    result.to_csv(TEST_OUTPUT, index=False)

    changed = int((corrected != given).sum())
    print(f"rewrote {changed} labels")
    print(f"wrote {TEST_OUTPUT}")
    print(f"wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
