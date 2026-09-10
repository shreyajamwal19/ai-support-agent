"""Check what fraction of brand replies are pure 'please DM us' deflections vs substantive
public resolutions -- critical for whether grounded retrieval is even meaningful."""
import pandas as pd, re, json
from pathlib import Path

RAW = Path("data/raw/twcs.csv")
df = pd.read_csv(RAW, dtype={"tweet_id":"int64","author_id":str}, low_memory=False)

deflect_pat = re.compile(r"\b(dm|direct message|private message|pm us|send us a message)\b", re.I)

candidates = ["AmazonHelp","AppleSupport","Uber_Support","SpotifyCares","Delta","AmericanAir","comcastcares"]
rows = []
for h in candidates:
    b = df[(df.inbound==False) & (df.author_id==h)]
    n = len(b)
    deflect = b["text"].str.contains(deflect_pat, na=False).sum()
    avg_words = b["text"].str.split().str.len().mean()
    rows.append({"brand":h, "n_replies":int(n), "pct_deflection_to_dm": round(100*deflect/n,1), "avg_reply_word_count": round(float(avg_words),1)})
out = pd.DataFrame(rows).sort_values("pct_deflection_to_dm")
print(out.to_string(index=False))
out.to_csv("artifacts/data_quality/deflection_rates.csv", index=False)
