import os
import json
import pandas as pd
from datetime import datetime

from build_seed import (
    canonical_team,
    date_to_ts,
    load_former_names,
    parse_bool,
    parse_date,
    sanitize_slug,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.environ.get("WC_DATA_DIR", f"{PROJECT_DIR}/data")
RAW_DATA_DIR = os.environ.get("WC_RAW_DATA_DIR", f"{DATA_DIR}/raw")
REFERENCE_DATA_DIR = os.environ.get("WC_REFERENCE_DATA_DIR", f"{DATA_DIR}/reference")
PROCESSED_DATA_DIR = os.environ.get("WC_PROCESSED_DATA_DIR", f"{DATA_DIR}/processed")


df = pd.read_csv(f"{RAW_DATA_DIR}/results.csv", sep=",")
former_names = load_former_names(f"{REFERENCE_DATA_DIR}/former_names.csv")
df = df[
    (pd.to_datetime(df['date']) >= datetime(1930, 1, 1)) &
    (pd.to_datetime(df['date']) < datetime(2026, 6, 11))
]


match_list = []
for _, row in df.iterrows():
    match_date = parse_date(row["date"])
    home_name = canonical_team(row["home_team"], match_date, former_names)
    away_name = canonical_team(row["away_team"], match_date, former_names)
    match_list.append({
        "date": row["date"],
        "ts": date_to_ts(match_date),
        "homeSlug": sanitize_slug(home_name),
        "awaySlug": sanitize_slug(away_name),
        "homeName": home_name,
        "awayName": away_name,
        "hg": int(row["home_score"]),
        "ag": int(row["away_score"]),
        "leagueName": row["tournament"],
        "neutral": parse_bool(row["neutral"]),
    })


os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
with open(f"{PROCESSED_DATA_DIR}/historical_results.json", "w", encoding='utf-8') as f:
    json.dump({"matches": match_list}, f, indent=4)

print(f"→ wrote to {PROCESSED_DATA_DIR}/historical_results.json")
