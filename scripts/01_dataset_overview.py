"""Phase 0/1: load raw TWCS csv, compute basic integrity + volume stats per brand candidate."""
import pandas as pd
import json
from pathlib import Path

RAW = Path("data/raw/twcs.csv")
OUT = Path("artifacts/data_quality")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, dtype={"tweet_id": "int64", "author_id": str, "in_response_to_tweet_id": "Int64"})
print("rows:", len(df))
print("cols:", df.columns.tolist())
print("nulls:\n", df.isna().sum())

# inbound=False rows authored by a brand handle; inbound=True are customers (author_id is a numeric hash)
brand_msgs = df[df["inbound"] == False]
brand_counts = brand_msgs["author_id"].value_counts()
print("\nTop 25 brand accounts by message volume:")
print(brand_counts.head(25))

overview = {
    "total_rows": int(len(df)),
    "n_unique_tweet_ids": int(df["tweet_id"].nunique()),
    "n_inbound_customer_msgs": int((df["inbound"] == True).sum()),
    "n_outbound_brand_msgs": int((df["inbound"] == False).sum()),
    "n_unique_brand_accounts": int(brand_msgs["author_id"].nunique()),
    "null_counts": df.isna().sum().to_dict(),
    "top_25_brand_accounts_by_volume": brand_counts.head(25).to_dict(),
    "date_min": str(df["created_at"].min()),
    "date_max_sample": str(df["created_at"].tail(1).values[0]),
}
with open(OUT / "overview.json", "w") as f:
    json.dump(overview, f, indent=2, default=str)
print("\nSaved overview.json")
