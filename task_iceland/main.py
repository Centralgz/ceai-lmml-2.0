import csv


def skip_preamble(lines):
    """Drop comment/blank preamble; header length varies between exports."""
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        start = i
        break
    return lines[start:]


def fix_station_name(name):
    """Undo UTF-8 bytes that were decoded as Latin-1 (mojibake)."""
    while True:
        try:
            nxt = name.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            break
        if nxt == name:
            break
        name = nxt
    return name


def parse_temperature(raw):
    return float(raw.replace(",", "."))


def to_celsius(temp, unit):
    if unit.strip().upper() == "F":
        return (temp - 32.0) * 5.0 / 9.0
    return temp


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.readlines()

    body = skip_preamble(lines)
    reader = csv.DictReader(body, delimiter=";")
    cleaned = {}
    for row in reader:
        reading_id = int(row["reading_id"].strip())
        station_name = fix_station_name(row["station_name"].strip())
        temperature_c = to_celsius(
            parse_temperature(row["temperature_raw"].strip()),
            row["unit"],
        )
        reading_date = row["reading_date"].strip()
        cleaned[reading_id] = {
            "reading_id": reading_id,
            "station_name": station_name,
            "temperature_c": temperature_c,
            "reading_date": reading_date,
        }
    return [cleaned[k] for k in sorted(cleaned)]


def write_output(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("reading_id,station_name,temperature_c,reading_date\n")
        for row in rows:
            temp = f"{row['temperature_c']:.1f}"
            f.write(
                f"{row['reading_id']},{row['station_name']},{temp},{row['reading_date']}\n"
            )


def main():
    rows = load_rows("test-input.csv")
    write_output("test-output.csv", rows)


if __name__ == "__main__":
    main()
