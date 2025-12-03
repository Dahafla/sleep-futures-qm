import pandas as pd
from dateutil import parser
from .config import DATA_RAW, TARGET_SLEEP_HOURS


def load_raw_sleep_data(path: str | None = None) -> pd.DataFrame:

    csv_path = path or DATA_RAW
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Parse datetimes
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    # Derive date (sleep "day" is the date of wake-up)
    df["date"] = df["end"].dt.date
    df["date"] = pd.to_datetime(df["date"])


    df["duration_minutes"] = df["duration"]
    df["hours_slept"] = df["duration_minutes"] / 60.0

    # Sleep efficiency
    restful = df.get("restful_minutes", 0)
    restless = df.get("restless_minutes", 0)
    awake = df.get("awake_minutes", 0)
    total_tracked = restful + restless + awake

    df["sleep_efficiency"] = restful / total_tracked.replace(0, pd.NA)

    # Bedtime in minutes after midnight
    df["bedtime_minutes"] = df["start"].dt.hour * 60 + df["start"].dt.minute

    # Sleep Index = hours_slept - TARGET_SLEEP_HOURS
    df["sleep_index"] = df["hours_slept"] - TARGET_SLEEP_HOURS

    # Sort & keep one row per day (if multiple, take last)
    df = df.sort_values("end").drop_duplicates("date", keep="last")

    # Set daily index
    df = df.set_index("date").sort_index()

    # reindex to full daily range (to see missing days)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    df.index.name = "date"

    return df[[
        "start",
        "end",
        "duration_minutes",
        "hours_slept",
        "sleep_efficiency",
        "bedtime_minutes",
        "sleep_index",
    ]]
