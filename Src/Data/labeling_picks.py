import pandas as pd
from pathlib import Path
import os

from Src.Data.fetch_data import stellar_data

folder_name=Path("Data/raw")
ebs=pd.read_csv("Data/eb_data.csv")
transits=pd.read_csv("Data/cumulative_koi_data.csv")

confirmed_set = set(transits.loc[transits["koi_disposition"] == "CONFIRMED", "kepid"].astype(str))
ebs_set = set(ebs["kepid"].astype(str))
stellar_set = set(stellar_data["kepid"].astype(str))

rows=[]
counts={'transit':0,'ebs':0, 'quiets':0}
eb_count=0
stellar_count=0
transit_count=0

for filepath in folder_name.glob("*.csv"):

    kepid=filepath.stem.replace("KIC","")

    if kepid in confirmed_set:
        counts['transit']+=1
        source="transit"
    elif kepid in ebs_set:
        counts['ebs']+=1
        source = "ebs"
    elif kepid in stellar_set:
        counts['quiets']+=1
        source = "quiets"



    if  counts[source]<=200:
        rows.append({"kepid":kepid, "source": source})
    else:
        os.remove(filepath)

df=pd.DataFrame(rows)
df.sort_values(by='source')

df.to_csv("Data/raw/_target_values.csv", index=False)





print(f"Manifest created successfully! Found {len(df)} light curves.")
print(df["source"].value_counts())  # This will print out exactly how many of each you have!